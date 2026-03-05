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

# E2E mode cost function weights - heavily weight model's predictions
E2E_X_EGO_COST = 5.0      # Higher weight on following model's position
E2E_V_EGO_COST = 10.0     # Higher weight on following model's velocity
E2E_A_EGO_COST = 8.0      # Higher weight on following model's acceleration
E2E_J_EGO_COST = 2.0      # Lower jerk cost to allow model's aggressive maneuvers

# Safety floor parameters - radar-based distance as minimum safe distance
SAFETY_FLOOR_MARGIN = 0.5  # Additional margin for safety floor
MIN_BRAKE_SAFETY_FACTOR = 1.2  # Model must brake at least this much before MPC overrides

# Lead transition filter parameters - smooth transitions to prevent phantom braking
LEAD_TRANSITION_FILTER_ALPHA = 0.3  # Smoothing factor for lead distance/velocity transitions
LEAD_SWITCH_HYSTERESIS = 2.0  # Minimum distance difference (m) to switch lead focus


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

    # E2E trajectory storage
    self.e2e_v_trajectory = np.zeros(CONTROL_N)
    self.e2e_a_trajectory = np.zeros(CONTROL_N)
    self.e2e_prob = 0.0
    self.e2e_valid = False

    # Lead transition filter - smooth transitions between leads to prevent phantom braking
    self.lead_d_rel_filtered = None
    self.lead_v_rel_filtered = None
    self.prev_lead_id = -1
    # Time constant for lead transition smoothing (seconds)
    self.lead_transition_time_constant = 0.3

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
    Update the longitudinal planner with E2E trajectory from the model.

    In ExperimentalMode, the MPC acts as a safety/jerk filter for the model's E2E output,
    heavily weighting the model's predicted velocity and acceleration rather than calculating
    targets based on radar/lead-car distance.

    Args:
      sm: SubMaster with current state
      e2e_x: E2E position trajectory (interpolated to MPC timesteps)
      e2e_v: E2E velocity trajectory (interpolated to MPC timesteps)
      e2e_a: E2E acceleration trajectory (interpolated to MPC timesteps)
      e2e_prob: Model confidence in the E2E trajectory
      e2e_valid: Whether the E2E trajectory is valid
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

    # Store E2E trajectory for use in MPC
    self.e2e_valid = e2e_valid and e2e_v is not None and e2e_a is not None
    self.e2e_prob = e2e_prob if e2e_prob is not None else 0.0

    if self.e2e_valid:
      self.e2e_v_trajectory = e2e_v
      self.e2e_a_trajectory = e2e_a

    # In ExperimentalMode, configure MPC to follow E2E trajectory
    is_experimental = sm['selfdriveState'].experimentalMode

    if is_experimental and self.e2e_valid:
      # Set E2E-specific cost weights - heavily weight model's predictions
      # In experimental mode, increase jerk penalty for smoother control
      self._set_e2e_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality,
                            experimental_mode=True)

      # Pass E2E trajectory to MPC for guidance
      self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality,
                           e2e_mode=True, e2e_v=self.e2e_v_trajectory, e2e_a=self.e2e_a_trajectory,
                           experimental_mode=True)
    else:
      # Standard mode - use traditional radar/lead-car based planning
      self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)

    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
                    e2e_mode=is_experimental and self.e2e_valid,
                    e2e_v=self.e2e_v_trajectory if self.e2e_valid else None,
                    e2e_a=self.e2e_a_trajectory if self.e2e_valid else None,
                    experimental_mode=is_experimental)

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
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    # Vision-only lead logic: when radar is unavailable, use modelV2.leadsV3
    lead_data = self._process_lead_data(sm, v_ego)

    # Safety floor: if lead car present, ensure we don't under-brake compared to radar-based distance
    if lead_data['status'] and self.e2e_valid:
      # Calculate minimum safe deceleration based on lead distance
      lead_d_rel = lead_data['dRel']
      lead_v_rel = lead_data['vRel']

      # Simple safety calculation: ensure we can stop before hitting lead
      if lead_d_rel > 0 and lead_v_rel < 0:  # Lead is closer and approaching
        min_safe_decel = (v_ego**2 - lead_v_rel**2) / (2 * lead_d_rel * SAFETY_FLOOR_MARGIN)
        min_safe_decel = max(min_safe_decel, ACCEL_MIN)

        # Only override if model is under-braking (not braking enough)
        if output_a_target_e2e > min_safe_decel:
          output_a_target_e2e = min_safe_decel * MIN_BRAKE_SAFETY_FACTOR

    if is_experimental:
      # In E2E mode, use model's acceleration with safety floor applied
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def _process_lead_data(self, sm, v_ego):
    """
    Process lead car data, prioritizing vision leads when radar is unavailable.
    Implements lead transition filtering to prevent phantom braking.

    Args:
      sm: SubMaster with current state
      v_ego: Ego vehicle speed

    Returns:
      Dictionary with lead status, dRel, and vRel
    """
    radar_unavailable = not sm['radarState'].leadOne.status and not sm['radarState'].leadTwo.status

    # Check for vision leads from modelV2.leadsV3
    leads_v3 = sm['modelV2'].leadsV3
    has_vision_leads = len(leads_v3) > 0 and leads_v3[0].prob > 0.5

    if radar_unavailable and has_vision_leads:
      # Use vision lead when radar is unavailable
      vision_lead = leads_v3[0]
      current_lead_id = 0  # Simple ID based on lead index

      # Apply lead transition filter to smooth switching between leads
      if self.prev_lead_id != current_lead_id and self.prev_lead_id != -1:
        # Lead switched - apply smoothing filter to prevent phantom braking
        if self.lead_d_rel_filtered is not None:
          # Smooth the transition using first-order filter
          d_rel_raw = float(vision_lead.x[0])
          self.lead_d_rel_filtered = (1 - LEAD_TRANSITION_FILTER_ALPHA) * self.lead_d_rel_filtered + LEAD_TRANSITION_FILTER_ALPHA * d_rel_raw

          v_rel_raw = float(vision_lead.v[0]) - v_ego
          if self.lead_v_rel_filtered is not None:
            self.lead_v_rel_filtered = (1 - LEAD_TRANSITION_FILTER_ALPHA) * self.lead_v_rel_filtered + LEAD_TRANSITION_FILTER_ALPHA * v_rel_raw
          else:
            self.lead_v_rel_filtered = v_rel_raw
        else:
          self.lead_d_rel_filtered = float(vision_lead.x[0])
          self.lead_v_rel_filtered = float(vision_lead.v[0]) - v_ego
      else:
        # No lead switch - use raw values or initialize filter
        self.lead_d_rel_filtered = float(vision_lead.x[0])
        self.lead_v_rel_filtered = float(vision_lead.v[0]) - v_ego

      self.prev_lead_id = current_lead_id

      return {
        'status': True,
        'dRel': self.lead_d_rel_filtered if self.lead_d_rel_filtered is not None else float(vision_lead.x[0]),
        'vRel': self.lead_v_rel_filtered if self.lead_v_rel_filtered is not None else float(vision_lead.v[0]) - v_ego,
        'source': 'vision'
      }
    elif sm['radarState'].leadOne.status:
      # Use radar lead when available - reset filter state
      self.prev_lead_id = -1
      self.lead_d_rel_filtered = None
      self.lead_v_rel_filtered = None

      return {
        'status': True,
        'dRel': sm['radarState'].leadOne.dRel,
        'vRel': sm['radarState'].leadOne.vLead - v_ego,
        'source': 'radar'
      }
    else:
      # No lead detected
      self.prev_lead_id = -1
      self.lead_d_rel_filtered = None
      self.lead_v_rel_filtered = None

      return {
        'status': False,
        'dRel': 0.0,
        'vRel': 0.0,
        'source': 'none'
      }

  def _set_e2e_weights(self, prev_accel_constraint=True, personality=None, experimental_mode=False):
    """
    Set cost weights for E2E mode where MPC acts as safety/jerk filter.

    In E2E mode, the MPC heavily weights following the model's predicted
    velocity and acceleration rather than calculating targets from radar.

    Args:
      prev_accel_constraint: Whether to penalize acceleration changes
      personality: Longitudinal personality setting
      experimental_mode: Whether in experimental mode (increases jerk penalty for smoother control)
    """
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import E2E_J_EGO_COST_EXPERIMENTAL

    jerk_factor = 1.0  # Use default jerk factor
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0

    # E2E-specific weights that prioritize following model predictions
    # In experimental mode, increase jerk penalty for smoother control
    e2e_j_cost = E2E_J_EGO_COST_EXPERIMENTAL if experimental_mode else E2E_J_EGO_COST
    cost_weights = [
      E2E_X_EGO_COST,    # Position cost - follow model's trajectory
      E2E_V_EGO_COST,    # Velocity cost - match model's velocity
      E2E_A_EGO_COST,    # Acceleration cost - match model's acceleration
      jerk_factor * a_change_cost,
      jerk_factor * e2e_j_cost  # Jerk cost - allow model's maneuvers
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

    # Include E2E-specific information in the plan message
    longitudinalPlan.e2eAcceleration = float(self.e2e_a_trajectory[0]) if self.e2e_valid else 0.0
    longitudinalPlan.modelConfidence = float(self.e2e_prob) if self.e2e_valid else 0.0

    pm.send('longitudinalPlan', plan_send)
