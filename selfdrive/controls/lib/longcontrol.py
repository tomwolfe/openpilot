import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


class LongControl:
  """
  Classical longitudinal controller using PID control.
  
  E2E Phase 2: This classical controller is deprecated in favor of LongControlE2E.
  The state machine has been simplified - LongCtrlState.pid handles the entire
  driving envelope including stopping. The model learns appropriate braking
  behavior for stops implicitly from human driving data.
  
  Removed states:
  - LongCtrlState.stopping: Model decides when to stop via brake_pred output
  - LongCtrlState.starting: Handled by normal PID control
  """
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """
    Update longitudinal control using PID.
    
    E2E Phase 2: should_stop parameter is ignored - the model's brake_pred
    output determines stopping behavior. This maintains backward compatibility
    with classical mode while removing heuristic stopping logic.
    """
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    if not active:
      self.reset()
      output_accel = 0.
      self.long_control_state = LongCtrlState.off
    else:
      # E2E Phase 2: Directly follow acceleration targets from the model/MPC.
      # The PID controller handles the tracking error for the entire driving envelope.
      # should_stop is ignored - model learns appropriate stopping behavior.
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target)
      
      # E2E Phase 2: Simplified state machine - only pid or off
      # Removed: LongCtrlState.stopping, LongCtrlState.starting
      self.long_control_state = LongCtrlState.pid

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
