#!/usr/bin/env python3
"""
E2E Phase 2: Direct Lateral Control

This controller bypasses the classical PID and torque controllers and directly applies
the model's predicted steering torque/angle to the actuators.

The neural network outputs steer_torque_pred and steer_angle_pred which are mapped
directly to CC.actuators.torque or CC.actuators.steeringAngleDeg.

E2E Architecture:
- No error calculation (model learns closed-loop control)
- No vehicle model dependencies (model learns vehicle dynamics implicitly)
- No comfort heuristics (model learns appropriate limits from data)
- Only safety-critical rate limiting (handled by Panda)
"""
import math
import numpy as np
from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.latcontrol import LatControl


class LatControlE2E(LatControl):
  """
  E2E Phase 2: Direct actuation lateral controller.

  Instead of calculating steering error and using a PID controller,
  this directly applies the model's steering torque predictions with minimal filtering.
  
  The neural network has learned:
  - Vehicle dynamics (steerRatio, mass, tire stiffness) implicitly from data
  - Appropriate comfort bounds (lateral accel, jerk) from human driving
  - Closed-loop control (compensating for errors, delays, disturbances)
  """

  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)

    # Low-pass filter for smoothing torque commands
    # Time constant tuned for smooth but responsive control
    # E2E Phase 2: Model outputs are already smooth from training, minimal filtering needed
    LAT_FILTER_SECONDS = 0.05
    self.torque_filter = FirstOrderFilter(0.0, LAT_FILTER_SECONDS, dt)
    self.angle_filter = FirstOrderFilter(0.0, LAT_FILTER_SECONDS, dt)

    # Control mode: prefer torque control for cars that support it
    # E2E Phase 2: Model predicts both torque and angle, car decides which to use
    self.control_mode = 'torque' if CP.steerControlType == car.CarParams.SteerControlType.torque else 'angle'

  def reset(self):
    """Reset controller state and filters."""
    self.torque_filter.x = 0.0
    self.angle_filter.x = 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay,
             model_output=None):
    """
    Update lateral control using direct E2E actuation.

    E2E Phase 2: The model directly predicts actuator commands.
    No curvature calculation, no vehicle model, no error feedback loop.
    
    Args:
      active: Whether lateral control is active
      CS: CarState message
      VM: VehicleModel instance (None for E2E - not needed)
      params: LiveParameters (None for E2E - not needed)
      steer_limited_by_safety: Whether steering is limited by safety system
      desired_curvature: Desired curvature from model (ignored in E2E)
      curvature_limited: Whether curvature is limited (ignored in E2E)
      lat_delay: Latency in seconds
      model_output: Dict containing model predictions including 'steer_torque_pred' and/or 'steer_angle_pred'

    Returns:
      output_torque: Torque command to send to actuators
      steering_angle_deg: Angle command (for angle control mode)
      pid_log: Log message for compatibility
    """
    pid_log = log.ControlsState.LateralE2EState.new_message()
    pid_log.version = 1

    output_torque = 0.0
    steering_angle_deg = 0.0

    # E2E Phase 2: Direct actuation from model predictions
    # No fallback to curvature-based control - model must always output predictions
    if model_output is not None:
      # Extract model predictions
      # Model outputs are in range [-1, 1] representing full left to full right
      steer_torque_pred = model_output.get('steer_torque_pred', 0.0)
      steer_angle_pred = model_output.get('steer_angle_pred', 0.0)

      # Apply low-pass filtering for smooth actuation
      # Filter removes high-frequency noise while preserving model's intended trajectory
      filtered_torque = self.torque_filter.update(steer_torque_pred)
      filtered_angle = self.angle_filter.update(steer_angle_pred)

      # Scale torque to [-1, 1] range expected by actuators
      # Panda safety layer enforces absolute limits
      output_torque = np.clip(filtered_torque, -1.0, 1.0)

      # Convert angle prediction to degrees for angle-control vehicles
      steering_angle_deg = math.degrees(filtered_angle)
    else:
      # E2E Phase 2: No model output is an error condition
      # Model should always output predictions when controls are active
      # This path should never be reached in normal operation
      cloudlog.warning("LatControlE2E: No model output received - this should not happen")

    if not active:
      self.reset()
      output_torque = 0.0
      steering_angle_deg = 0.0
      pid_log.active = False
    else:
      pid_log.active = True
      pid_log.steeringAngleDesiredDeg = steering_angle_deg
      pid_log.outputTorque = float(output_torque)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return output_torque, steering_angle_deg, pid_log
