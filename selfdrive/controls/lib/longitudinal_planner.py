#!/usr/bin/env python3
"""
Unified Neural Execution - Longitudinal Planner

This planner uses E2E (End-to-End) neural model outputs as the primary input
for longitudinal control. Classical MPC is deprecated and only used as a fallback
when ForceClassicalMPC parameter is enabled.
"""
import math
import numpy as np

import cereal.messaging as messaging
from cereal import log as cereal_log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX, V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.common.swaglog import cloudlog

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# E2E longitudinal personality gains and offsets
# Applied to model output to adjust aggressiveness
PERSONALITY_GAINS = {
  cereal_log.LongitudinalPersonality.relaxed: 0.85,    # Gentler acceleration/braking
  cereal_log.LongitudinalPersonality.standard: 1.0,    # No modification
  cereal_log.LongitudinalPersonality.aggressive: 1.15, # More responsive
}

# Personality-based acceleration offsets (m/s^2)
PERSONALITY_OFFSETS = {
  cereal_log.LongitudinalPersonality.relaxed: -0.2,    # Slightly less acceleration
  cereal_log.LongitudinalPersonality.standard: 0.0,
  cereal_log.LongitudinalPersonality.aggressive: 0.3,  # More acceleration
}

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


def apply_personality_to_accel(a_target, personality):
  """Apply personality gain and offset to acceleration command."""
  gain = PERSONALITY_GAINS.get(personality, 1.0)
  offset = PERSONALITY_OFFSETS.get(personality, 0.0)
  return a_target * gain + offset


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.dt = dt
    self.allow_throttle = True
    self.e2e_active = True
    
    # Check if classical MPC should be forced (legacy mode)
    # Default to False (E2E mode) if param doesn't exist yet
    params = Params()
    try:
      self.force_classical_mpc = params.get_bool("ForceClassicalMPC")
    except Exception:
      # Param not registered yet - default to E2E mode
      self.force_classical_mpc = False
    
    # Only initialize MPC if in legacy mode
    if self.force_classical_mpc:
      self.mpc = LongitudinalMpc(dt=dt)
      cloudlog.warning("ForceClassicalMPC enabled - using legacy MPC controller")
    else:
      self.mpc = None
      cloudlog.info("E2E longitudinal active.")
    
    self.fcw = False

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update_e2e_longitudinal(self, sm, personality):
    """
    Update longitudinal plan using E2E model outputs directly.
    
    The model provides:
    - modelV2.action.desiredAcceleration: immediate acceleration target
    - modelV2.action.shouldStop: stopping flag
    - modelV2.longitudinalPlan.speeds/accels: trajectory over time horizon
    """
    # Get E2E model outputs
    e2e_accel = sm['modelV2'].action.desiredAcceleration
    e2e_should_stop = sm['modelV2'].action.shouldStop
    
    # Apply personality adjustments
    adjusted_accel = apply_personality_to_accel(e2e_accel, personality)
    
    # Extract trajectory from model if available
    # Note: modelV2.longitudinalPlan would be populated by the model
    # For now, we use the action output and construct a simple trajectory
    v_ego = sm['carState'].vEgo
    
    # Use model's desired acceleration and project trajectory
    # This is a simplified approach - in production, the model would output
    # the full trajectory directly
    self.a_desired_trajectory[0] = adjusted_accel
    for i in range(1, CONTROL_N):
      # Simple integration for trajectory projection
      dt_step = CONTROL_N_T_IDX[i] - CONTROL_N_T_IDX[i-1]
      prev_v = self.v_desired_trajectory[i-1] if i > 0 else v_ego
      self.a_desired_trajectory[i] = adjusted_accel
      self.v_desired_trajectory[i] = prev_v + adjusted_accel * dt_step
    
    self.v_desired_trajectory[0] = v_ego
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    
    self.output_a_target = adjusted_accel
    self.output_should_stop = e2e_should_stop
    self.e2e_active = True
    
    return self.output_a_target, self.output_should_stop

  def update_mpc_longitudinal(self, sm, v_cruise, personality):
    """Legacy MPC-based longitudinal control (deprecated)."""
    v_ego = sm['carState'].vEgo
    
    # Reset current state when not engaged
    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized
    
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)
    
    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    
    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target, output_should_stop = get_accel_from_plan(
      self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
      action_t=action_t, vEgoStopping=self.CP.vEgoStopping
    )
    
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.output_should_stop = output_should_stop
    self.e2e_active = False
    
    return self.output_a_target, self.output_should_stop

  def update(self, sm):
    """
    Main update loop - dispatches to E2E or MPC based on ForceClassicalMPC param.
    
    E2E Mode (default):
      - Uses modelV2.action.desiredAcceleration as primary input
      - Applies personality-based gain/offset
      - No MPC solver overhead
    
    Legacy MPC Mode (ForceClassicalMPC=True):
      - Uses classical radar-centric MPC
      - Maintains backward compatibility
    """
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    
    personality = sm['selfdriveState'].personality
    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if self.force_classical_mpc:
      # Legacy MPC mode
      self.update_mpc_longitudinal(sm, v_cruise, personality)
    else:
      # E2E mode (default)
      self.update_e2e_longitudinal(sm, personality)
      
      # Update allow_throttle from model predictions
      _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
      self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED
      
      if not self.allow_throttle:
        clipped_accel_coast = max(accel_coast, accel_clip[0])
        clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], 
                                                [accel_clip[1], clipped_accel_coast])
        accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Apply acceleration clipping with rate limiting
    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, 
                                 self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(self.output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    
    # Only set solver time if MPC is active (legacy mode)
    if self.mpc is not None:
      longitudinalPlan.solverExecutionTime = self.mpc.solve_time
    else:
      longitudinalPlan.solverExecutionTime = 0.0

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = (
      LongitudinalPlanSource.e2e if self.e2e_active else LongitudinalPlanSource.cruise
    )
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
