#!/usr/bin/env python3
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.modeld.constants import ModelConstants
import cereal.messaging as messaging
import numpy as np


def extract_optimal_path(model_v2_msg):
  """
  Extract the best trajectory from the model's multi-hypothesis predictions.

  Uses the PLAN_MHP_N and PLAN_MHP_SELECTION constants to identify the hypothesis
  with the highest probability. Returns the selected trajectory's position, velocity,
  acceleration, and confidence level.

  Args:
    model_v2_msg: The modelV2 message containing policy hypotheses

  Returns:
    Tuple of (position_x, velocity_x, acceleration_x, probability, is_valid)
    where each array is interpolated to MPC timesteps, or None values if no valid policy
  """
  # E2E Phase 1: MPC time indices for trajectory interpolation
  # TODO: Migrate to direct E2E trajectory without MPC interpolation
  LONG_MPC_N = 12
  LONG_MPC_MAX_T = 10.0
  T_IDXS_MPC = np.array([index_function(idx, max_val=LONG_MPC_MAX_T, max_idx=LONG_MPC_N) for idx in range(LONG_MPC_N + 1)])

  # Check if policy hypotheses are available
  if not model_v2_msg.policy or len(model_v2_msg.policy) == 0:
    # Fallback to legacy single-hypothesis output if policy is not available
    if (len(model_v2_msg.position.x) == ModelConstants.IDX_N and
        len(model_v2_msg.velocity.x) == ModelConstants.IDX_N and
        len(model_v2_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_v2_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_v2_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_v2_msg.acceleration.x)
      return x, v, a, 1.0, True
    return None, None, None, 0.0, False

  # Find the hypothesis with the highest probability
  best_hypothesis = None
  best_probability = -1.0

  for hypothesis in model_v2_msg.policy:
    if hypothesis.probability > best_probability:
      best_probability = hypothesis.probability
      best_hypothesis = hypothesis

  if best_hypothesis is None:
    return None, None, None, 0.0, False

  # Extract trajectory data from the best hypothesis
  # Interpolate to MPC timesteps for consistency with longitudinal planner
  if len(best_hypothesis.trajectory.x) == ModelConstants.IDX_N:
    x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, best_hypothesis.trajectory.x)
    v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, best_hypothesis.velocity.x)
    a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, best_hypothesis.acceleration.x)
    return x, v, a, best_probability, True

  return None, None, None, 0.0, False


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  longitudinal_planner = LongitudinalPlanner(CP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState'],
                           poll='modelV2')

  while True:
    sm.update()
    if sm.updated['modelV2']:
      # Extract optimal path from multi-hypothesis predictions
      e2e_x, e2e_v, e2e_a, e2e_prob, e2e_valid = extract_optimal_path(sm['modelV2'])

      # Pass E2E trajectory to longitudinal planner
      longitudinal_planner.update(sm, e2e_x=e2e_x, e2e_v=e2e_v, e2e_a=e2e_a,
                                   e2e_prob=e2e_prob, e2e_valid=e2e_valid)
      longitudinal_planner.publish(sm, pm)

      # E2E Phase 4: Lane departure warning from model (not heuristic)
      # Model outputs single ldwWarning flag - use for both left and right
      # Future: model can be enhanced to output separate left/right warnings
      ldw_warning = sm['modelV2'].action.ldwWarning
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw_warning
      msg.driverAssistance.rightLaneDeparture = ldw_warning
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
