#!/usr/bin/env python3
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.modeld.constants import ModelConstants
import cereal.messaging as messaging
import numpy as np


def extract_optimal_path(model_v2_msg):
  """
  Extract the best trajectory from the model's multi-hypothesis predictions.

  Phase 3: Direct Longitudinal Control

  Uses the PLAN_MHP_N and PLAN_MHP_SELECTION constants to identify the hypothesis
  with the highest probability. Returns the selected trajectory's position, velocity,
  acceleration, and confidence level.

  The extracted acceleration is used as the primary input for the MPC, making
  E2E longitudinal control the default behavior (Hybrid E2E mode).

  Args:
    model_v2_msg: The modelV2 message containing policy hypotheses

  Returns:
    Tuple of (position_x, velocity_x, acceleration_x, probability, is_valid)
    where each array is interpolated to MPC timesteps, or None values if no valid policy
  """
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC

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

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState'],
                           poll='modelV2')

  while True:
    sm.update()
    if sm.updated['modelV2']:
      # Phase 3: Extract optimal path from multi-hypothesis predictions
      # This trajectory is used as the primary input for Hybrid E2E longitudinal control
      e2e_x, e2e_v, e2e_a, e2e_prob, e2e_valid = extract_optimal_path(sm['modelV2'])

      # Pass E2E trajectory to longitudinal planner
      # The planner uses model acceleration as primary (Hybrid E2E by default)
      longitudinal_planner.update(sm, e2e_x=e2e_x, e2e_v=e2e_v, e2e_a=e2e_a,
                                   e2e_prob=e2e_prob, e2e_valid=e2e_valid)
      longitudinal_planner.publish(sm, pm)

      # LDW operates on model's spatial outputs - preserved from original implementation
      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
