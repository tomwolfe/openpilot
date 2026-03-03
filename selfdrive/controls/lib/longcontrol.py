from typing import Optional

import numpy as np
from cereal import car, log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants, PolicyType

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill):
  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state


class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    # E2E policy buffer for 5-second horizon (100Hz = 500 points)
    self.e2e_accel_buffer = np.zeros(ModelConstants.POLICY_HORIZON_POINTS, dtype=np.float32)
    self.e2e_buffer_idx = 0
    self.e2e_valid = False

  def reset(self):
    self.pid.reset()
    self.e2e_accel_buffer = np.zeros(ModelConstants.POLICY_HORIZON_POINTS, dtype=np.float32)
    self.e2e_buffer_idx = 0
    self.e2e_valid = False

  def update_e2e_policy(self, policy_msg: Optional[log.ModelDataV2.Policy]):
    """
    Update E2E policy from model output.
    
    In Phase 2 Full E2E, the model directly outputs longitudinal accelerations
    and lateral curvatures for a 5-second horizon, independent of explicit
    lead car or lane line detection.
    
    Args:
      policy_msg: Full E2E Policy message from modelV2.fullE2EPolicy
    """
    if policy_msg is None or not policy_msg.longitudinalAccelerations:
      self.e2e_valid = False
      return
    
    # Validate policy type - only accept standard, stopping, or starting policies
    valid_policy_types = {
      log.ModelDataV2.Policy.PolicyType.standard,
      log.ModelDataV2.Policy.PolicyType.stopping,
      log.ModelDataV2.Policy.PolicyType.starting,
    }
    
    if policy_msg.policyType not in valid_policy_types:
      self.e2e_valid = False
      return
    
    # Store acceleration profile
    accel_list = list(policy_msg.longitudinalAccelerations)
    if len(accel_list) >= ModelConstants.POLICY_HORIZON_POINTS:
      self.e2e_accel_buffer = np.array(accel_list[:ModelConstants.POLICY_HORIZON_POINTS], dtype=np.float32)
      self.e2e_buffer_idx = 0
      self.e2e_valid = True

  def get_e2e_accel(self, v_ego: float) -> float:
    """
    Get acceleration from E2E policy buffer.
    
    Returns the next acceleration value from the model's 5-second horizon.
    Falls back to 0.0 if E2E policy is invalid.
    
    Args:
      v_ego: Current ego velocity (for validation)
    
    Returns:
      Acceleration command from E2E policy
    """
    if not self.e2e_valid or self.e2e_buffer_idx >= len(self.e2e_accel_buffer):
      return 0.0
    
    accel = self.e2e_accel_buffer[self.e2e_buffer_idx]
    self.e2e_buffer_idx += 1
    return float(accel)

  def update(self, active, CS, a_target, should_stop, accel_limits, e2e_policy=None):
    """
    Update longitudinal control with optional E2E policy.
    
    In Phase 2 Full E2E, when E2E policy is available and valid, the system
    uses the model's direct acceleration output as the primary control signal.
    The PID controller acts as a fallback and for fine-tuning.
    
    Args:
      active: Whether longitudinal control is active
      CS: CarState message
      a_target: Target acceleration from MPC (fallback)
      should_stop: Whether the vehicle should stop
      accel_limits: Acceleration limits [min, max]
      e2e_policy: Optional E2E Policy message from model
    
    Returns:
      Acceleration command
    """
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    
    # Update E2E policy if provided
    if e2e_policy is not None:
      self.update_e2e_policy(e2e_policy)
    
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      # Check if E2E policy has a starting maneuver
      if self.e2e_valid and e2e_policy and e2e_policy.policyType == log.ModelDataV2.Policy.PolicyType.starting:
        output_accel = self.get_e2e_accel(CS.vEgo)
      else:
        output_accel = self.CP.startAccel
      # Don't reset buffer when starting - continue using E2E policy

    else:  # LongCtrlState.pid
      # Phase 2 Full E2E: Use model's direct acceleration output as primary
      if self.e2e_valid:
        e2e_accel = self.get_e2e_accel(CS.vEgo)
        # Use E2E acceleration with small PID correction for tracking error
        error = e2e_accel - CS.aEgo
        pid_correction = self.pid.update(error, speed=CS.vEgo, feedforward=e2e_accel)
        output_accel = e2e_accel + 0.1 * pid_correction  # Light PID correction
      else:
        # Fallback to traditional PID control
        error = a_target - CS.aEgo
        output_accel = self.pid.update(error, speed=CS.vEgo,
                                       feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
