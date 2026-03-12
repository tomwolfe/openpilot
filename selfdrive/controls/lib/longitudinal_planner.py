#!/usr/bin/env python3
"""
E2E Longitudinal Planner

DEPRECATED: This module previously used MPC for longitudinal planning.
As of E2E Phase 2, it now directly passes through model predictions.
The MPC libraries are kept for backward compatibility but are not used.

The neural network in modeld now outputs direct acceleration commands
which are used by controlsd.py's LongControl (E2E controller).
"""
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import index_function

# E2E Phase 1: MPC time indices (deprecated, kept for backward compatibility)
LONG_MPC_N = 12
LONG_MPC_MAX_T = 10.0
T_IDXS_MPC = np.array([index_function(idx, max_val=LONG_MPC_MAX_T, max_idx=LONG_MPC_N) for idx in range(LONG_MPC_N + 1)])

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# E2E mode cost function weights - DEPRECATED: MPC no longer used
# Kept for backward compatibility only
E2E_X_EGO_COST = 5.0
E2E_V_EGO_COST = 10.0
E2E_A_EGO_COST = 8.0
E2E_J_EGO_COST = 2.0

# Safety floor parameters - DEPRECATED: Model handles safety via crashProb/ttcPred
SAFETY_FLOOR_MARGIN = 0.5
MIN_BRAKE_SAFETY_FACTOR = 1.2

# Lead transition filter parameters - DEPRECATED: Vision-based lead detection only
LEAD_TRANSITION_FILTER_ALPHA = 0.3
LEAD_SWITCH_HYSTERESIS = 2.0


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
  """
  E2E Phase 2: Direct longitudinal planner.

  DEPRECATED: This class previously initialized an MPC for longitudinal planning.
  As of E2E Phase 2, it directly passes through model predictions without MPC.
  The MPC initialization is kept for backward compatibility but not used.
  """
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    # E2E: MPC deprecated - not initialized
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

    # E2E trajectory storage (for logging)
    self.e2e_v_trajectory = np.zeros(CONTROL_N)
    self.e2e_a_trajectory = np.zeros(CONTROL_N)
    self.e2e_prob = 0.0
    self.e2e_valid = False

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
    E2E Phase 2: Direct longitudinal planning from model predictions.

    The model now outputs direct acceleration commands via modelV2.action.desiredAcceleration.
    This planner passes through the model's predictions with minimal filtering.

    The MPC is deprecated and no longer used for primary control.
    Radar is used for safety redundancy only, not primary gap keeping.

    Args:
      sm: SubMaster with current state
      e2e_x: E2E position trajectory (for logging)
      e2e_v: E2E velocity trajectory (for logging)
      e2e_a: E2E acceleration trajectory (for logging)
      e2e_prob: Model confidence (for logging)
      e2e_valid: Whether E2E trajectory is valid (for logging)
    """
    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
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

    if not self.allow_throttle:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1]) if len(sm['carControl'].orientationNED) == 3 else ACCEL_MAX
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    # Store E2E trajectory for logging
    self.e2e_valid = e2e_valid and e2e_v is not None and e2e_a is not None
    self.e2e_prob = e2e_prob if e2e_prob is not None else 0.0
    if self.e2e_valid:
      self.e2e_v_trajectory = e2e_v
      self.e2e_a_trajectory = e2e_a

    # E2E Phase 2: Direct acceleration from model
    # The model learns appropriate acceleration targets from human driving data
    # No MPC, no radar-based gap keeping - pure vision-based control
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = False  # Model controls braking directly

    # Clip to vehicle limits
    output_a_target = np.clip(output_a_target_e2e, accel_clip[0], accel_clip[1])
    self.output_a_target = output_a_target
    self.output_should_stop = output_should_stop_e2e

    # Update desired state for next iteration
    self.a_desired = output_a_target
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * output_a_target

    # Generate trajectory for logging (simple integration of acceleration)
    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N - 1)

    v_current = v_ego
    for i in range(CONTROL_N):
      self.a_desired_trajectory[i] = output_a_target
      self.v_desired_trajectory[i] = v_current
      if i < CONTROL_N - 1:
        v_current += output_a_target * DT_MDL
        self.j_desired_trajectory[i] = 0.0  # Constant acceleration = zero jerk

    # E2E: No FCW from MPC - model handles collision avoidance via crashProb/ttcPred
    self.fcw = False

  def publish(self, sm, pm):
    """Publish longitudinal plan - E2E Phase 2: Direct model passthrough."""
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    # E2E: No MPC solver - set to 0
    longitudinalPlan.solverExecutionTime = 0.0

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    # E2E: Always use e2e source since we're using direct model output
    longitudinalPlan.longitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource.e2e
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    # Include E2E-specific information in the plan message
    longitudinalPlan.e2eAcceleration = float(self.e2e_a_trajectory[0]) if self.e2e_valid else 0.0
    longitudinalPlan.modelConfidence = float(self.e2e_prob) if self.e2e_valid else 0.0

    pm.send('longitudinalPlan', plan_send)
