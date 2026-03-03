#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  A_CHANGE_COST, LIMIT_COST, DANGER_ZONE_COST
)
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Longitudinal 1.0: Vision-based E2E is now the default
# Cost function weights for vision-longitudinal mode - heavily weight model's predictions
VISION_X_EGO_COST = 5.0      # Higher weight on following model's position
VISION_V_EGO_COST = 10.0     # Higher weight on following model's velocity
VISION_A_EGO_COST = 8.0      # Higher weight on following model's acceleration
VISION_J_EGO_COST = 2.0      # Lower jerk cost to allow model's aggressive maneuvers

# Classical lead-following weights (used as fallback or safety constraint)
CLASSIC_X_EGO_OBSTACLE_COST = 3.0
CLASSIC_X_EGO_COST = 0.0
CLASSIC_V_EGO_COST = 0.0
CLASSIC_A_EGO_COST = 0.0
CLASSIC_J_EGO_COST = 5.0

# Safety floor parameters - radar-based distance as minimum safe distance
SAFETY_FLOOR_MARGIN = 0.5  # Additional margin for safety floor
MIN_BRAKE_SAFETY_FACTOR = 1.2  # Model must brake at least this much before MPC overrides

# Model certainty thresholds for fallback to classical mode
MODEL_CERTAINTY_THRESHOLD = 0.3  # Below this, use classical lead-following
MODEL_CERTAINTY_HYSTERESIS = 0.1  # Hysteresis to prevent rapid switching


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


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

    # Vision-longitudinal trajectory storage
    self.vision_v_trajectory = np.zeros(CONTROL_N)
    self.vision_a_trajectory = np.zeros(CONTROL_N)
    self.vision_prob = 0.0
    self.vision_valid = False

    # Model certainty tracking for fallback logic
    self.using_vision_longitudinal = True  # Default to vision-based
    self.model_certainty_low_counter = 0  # Hysteresis counter

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

  def update(self, sm, e2e_x=None, e2e_v=None, e2e_a=None, e2e_prob=0.0, e2e_valid=False):
    """
    Update the longitudinal planner with vision-based trajectory from the model.

    Longitudinal 1.0: Vision-based E2E is now the default for all modes.
    The MPC acts as a safety/jerk filter for the model's output,
    heavily weighting the model's predicted velocity and acceleration rather than calculating
    targets based on radar/lead-car distance.

    Safety fallback: If model certainty is low, revert to classical lead-following behavior.

    Args:
      sm: SubMaster with current state
      e2e_v: Vision velocity trajectory (interpolated to MPC timesteps)
      e2e_a: Vision acceleration trajectory (interpolated to MPC timesteps)
      e2e_prob: Model confidence in the vision trajectory
      e2e_valid: Whether the vision trajectory is valid
    """
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    # Store vision trajectory for use in MPC
    self.vision_valid = e2e_valid and e2e_v is not None and e2e_a is not None
    self.vision_prob = e2e_prob if e2e_prob is not None else 0.0

    if self.vision_valid:
      self.vision_v_trajectory = e2e_v
      self.vision_a_trajectory = e2e_a

    # Determine if we should use vision-longitudinal or fall back to classical
    # Check model certainty with hysteresis to prevent rapid switching
    model_certainty_high = self.vision_prob > (MODEL_CERTAINTY_THRESHOLD + MODEL_CERTAINTY_HYSTERESIS)
    model_certainty_low = self.vision_prob < MODEL_CERTAINTY_THRESHOLD

    if model_certainty_high:
      self.model_certainty_low_counter = 0
    elif model_certainty_low:
      self.model_certainty_low_counter += 1

    # Require sustained low certainty before switching to classical (hysteresis)
    use_vision_longitudinal = self.vision_valid and (self.model_certainty_low_counter < 5)

    self.using_vision_longitudinal = use_vision_longitudinal

    if use_vision_longitudinal:
      # Vision-longitudinal mode: heavily weight model's predictions
      self._set_vision_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)

      # Pass vision trajectory to MPC for guidance
      self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality,
                           vision_mode=True, vision_v=self.vision_v_trajectory, vision_a=self.vision_a_trajectory)
    else:
      # Classical mode - use traditional radar/lead-car based planning
      self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)

    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
                    vision_mode=use_vision_longitudinal,
                    vision_v=self.vision_v_trajectory if self.vision_valid else None,
                    vision_a=self.vision_a_trajectory if self.vision_valid else None)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_vision = sm['modelV2'].action.desiredAcceleration
    output_should_stop_vision = sm['modelV2'].action.shouldStop

    # Safety floor: if lead car present, ensure we don't under-brake compared to radar-based distance
    if sm['radarState'].leadOne.status and self.vision_valid:
      # Calculate minimum safe deceleration based on radar distance
      lead_d_rel = sm['radarState'].leadOne.dRel
      lead_v_rel = sm['radarState'].leadOne.vLead - v_ego

      # Simple safety calculation: ensure we can stop before hitting lead
      if lead_d_rel > 0 and lead_v_rel < 0:  # Lead is closer and approaching
        min_safe_decel = (v_ego**2 - lead_v_rel**2) / (2 * lead_d_rel * SAFETY_FLOOR_MARGIN)
        min_safe_decel = max(min_safe_decel, ACCEL_MIN)

        # Only override if model is under-braking (not braking enough)
        if output_a_target_vision > min_safe_decel:
          output_a_target_vision = min_safe_decel * MIN_BRAKE_SAFETY_FACTOR

    if use_vision_longitudinal:
      # In vision-longitudinal mode, use model's acceleration with safety floor applied
      output_a_target = min(output_a_target_vision, output_a_target_mpc)
      self.output_should_stop = output_should_stop_vision or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      # Classical lead-following mode
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc
      self.mpc.source = LongitudinalPlanSource.lead0 if sm['radarState'].leadOne.status else LongitudinalPlanSource.cruise

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def _set_vision_weights(self, prev_accel_constraint=True, personality=None):
    """
    Set cost weights for vision-longitudinal mode where MPC acts as safety/jerk filter.

    In vision-longitudinal mode, the MPC heavily weights following the model's predicted
    velocity and acceleration rather than calculating targets from radar.
    """
    jerk_factor = 1.0  # Use default jerk factor
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0

    # Vision-longitudinal weights that prioritize following model predictions
    # Cost weights: [obstacle_dist, x_ego, v_ego, a_ego, a_change, jerk]
    cost_weights = [
      VISION_X_EGO_COST,    # Position cost - follow model's trajectory
      VISION_X_EGO_COST,    # x_ego cost
      VISION_V_EGO_COST,    # Velocity cost - match model's velocity
      VISION_A_EGO_COST,    # Acceleration cost - match model's acceleration
      jerk_factor * a_change_cost,  # a_change cost
      jerk_factor * VISION_J_EGO_COST  # Jerk cost - allow model's maneuvers
    ]
    constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self.mpc.set_cost_weights(cost_weights, constraint_cost_weights)

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    # Include vision-longitudinal-specific information in the plan message
    longitudinalPlan.e2eAcceleration = float(self.vision_a_trajectory[0]) if self.vision_valid else 0.0
    longitudinalPlan.modelConfidence = float(self.vision_prob) if self.vision_valid else 0.0
    longitudinalPlan.usingVisionLongitudinal = self.using_vision_longitudinal

    pm.send('longitudinalPlan', plan_send)
