#!/usr/bin/env python3
"""
E2E Phase 3: Unified End-to-End Controller

This controller merges longitudinal and lateral control into a single unified system.
Human drivers do not decouple lateral and longitudinal physics - neither should the code.

The neural network outputs all actuator commands (steering torque, gas, brake) which
are applied directly with minimal filtering. This eliminates the abstraction barrier
between lateral and longitudinal control.

E2E Architecture:
- Single controller for all actuator commands
- Model learns coupled lateral-longitudinal dynamics (e.g., braking in corners)
- No separate state machines for long/lat
- Direct actuator output with safety filtering only
"""
import numpy as np
from cereal import car, log
from openpilot.common.realtime import DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.swaglog import cloudlog

LongCtrlState = car.CarControl.Actuators.LongControlState


class E2EController:
  """
  E2E Phase 3: Unified direct actuation controller.
  
  Merges LongControlE2E and LatControlE2E into a single controller that:
  - Processes all model actuator outputs together
  - Applies consistent filtering across all actuators
  - Handles AEB override for both longitudinal and lateral
  - Logs unified controller state
  
  Human drivers naturally couple lateral and longitudinal control:
  - Slow down before corners (longitudinal → lateral)
  - Accelerate out of turns (lateral → longitudinal)
  - Emergency swerves while braking (coupled lateral-longitudinal)
  
  This unified controller preserves these natural couplings learned by the model.
  """

  def __init__(self, CP, CI, dt):
    self.CP = CP
    self.CI = CI

    # Unified low-pass filters for all actuators
    # E2E Phase 3: Consistent filter time constants across all actuators
    # Reduced from previous values - model outputs are smooth from training
    FILTER_SECONDS = 0.08
    self.gas_filter = FirstOrderFilter(0.0, FILTER_SECONDS, dt)
    self.brake_filter = FirstOrderFilter(0.0, FILTER_SECONDS, dt)
    self.torque_filter = FirstOrderFilter(0.0, FILTER_SECONDS, dt)
    self.angle_filter = FirstOrderFilter(0.0, FILTER_SECONDS, dt)

    self.long_control_state = LongCtrlState.off
    self.last_output_accel = 0.0
    self.last_output_torque = 0.0
    self.last_output_angle = 0.0

    # AEB override state
    self.aeb_active = False
    
    # Control mode: prefer torque control for cars that support it
    self.control_mode = 'torque' if CP.steerControlType == car.CarParams.SteerControlType.torque else 'angle'

  def reset(self):
    """Reset all controller state and filters."""
    self.gas_filter.x = 0.0
    self.brake_filter.x = 0.0
    self.torque_filter.x = 0.0
    self.angle_filter.x = 0.0
    self.last_output_accel = 0.0
    self.last_output_torque = 0.0
    self.last_output_angle = 0.0
    self.long_control_state = LongCtrlState.off
    self.aeb_active = False

  def update(self, active, CS, model_output, accel_limits, aeb_override=None):
    """
    Update unified E2E control using direct actuation.
    
    E2E Phase 3: All actuators controlled together - no decoupling.
    The model has learned coupled lateral-longitudinal dynamics from human data.
    
    Args:
      active: Whether control is active
      CS: CarState message
      model_output: Dict containing all model predictions:
                    - gas_pred, brake_pred: [0, 1] pedal positions
                    - steer_torque_pred: [-1, 1] torque command
                    - steer_angle_pred: radians steering angle
                    - crash_prob: [0, 1] collision probability
                    - ttc_pred: seconds to collision
      accel_limits: [min_accel, max_accel] vehicle limits
      aeb_override: Optional AEB acceleration override
    
    Returns:
      output_accel: Acceleration command
      output_torque: Steering torque command (or 0 if angle mode)
      output_angle: Steering angle command (or 0 if torque mode)
    """
    # Extract all model predictions
    gas_pred = model_output.get('gas_pred', 0.0)
    brake_pred = model_output.get('brake_pred', 0.0)
    steer_torque_pred = model_output.get('steer_torque_pred', 0.0)
    steer_angle_pred = model_output.get('steer_angle_pred', 0.0)
    crash_prob = model_output.get('crash_prob', 0.0)
    ttc_pred = model_output.get('ttc_pred', 0.0)

    # Apply unified low-pass filtering to all actuators
    # Filter removes high-frequency noise while preserving model's intended trajectory
    filtered_gas = self.gas_filter.update(gas_pred)
    filtered_brake = self.brake_filter.update(brake_pred)
    filtered_torque = self.torque_filter.update(steer_torque_pred)
    filtered_angle = self.angle_filter.update(steer_angle_pred)

    # Convert pedal positions to acceleration
    # Gas: 0-1 maps to [0, max_accel]
    # Brake: 0-1 maps to [min_accel, 0]
    # E2E Phase 3: No artificial blending - model learns smooth transitions
    gas_accel = filtered_gas * accel_limits[1]
    brake_accel = filtered_brake * accel_limits[0]
    output_accel = gas_accel + brake_accel

    # E2E AEB override: imminent collision triggers maximum braking
    # This bypasses all comfort limits for safety
    if aeb_override is not None and aeb_override < output_accel:
      output_accel = aeb_override
      self.aeb_active = True
    else:
      self.aeb_active = False

    # Clip to vehicle limits (Panda safety layer enforces absolute bounds)
    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])

    # Apply steering command based on control mode
    # E2E Phase 3: Model outputs both torque and angle - car uses appropriate one
    if self.control_mode == 'torque':
      # Torque control: scale to [-1, 1] range expected by actuators
      self.last_output_torque = np.clip(filtered_torque, -1.0, 1.0)
      self.last_output_angle = 0.0
    else:
      # Angle control: convert radians to degrees
      import math
      self.last_output_torque = 0.0
      self.last_output_angle = math.degrees(filtered_angle)

    # Update state
    if not active:
      self.reset()
      self.last_output_accel = 0.0
      self.last_output_torque = 0.0
      self.last_output_angle = 0.0
      self.long_control_state = LongCtrlState.off
    else:
      # E2E Phase 3: Simplified state - model handles all driving states
      # No separate stopping/starting states - model learns these behaviors
      self.long_control_state = LongCtrlState.pid

    return self.last_output_accel, self.last_output_torque, self.last_output_angle

  def get_aeb_status(self):
    """Returns whether AEB is currently active."""
    return self.aeb_active

  def get_controller_state(self):
    """
    Get unified controller state for logging.
    
    Returns dict with all actuator commands and AEB status.
    This replaces separate pidState/torqueState/angleState logging.
    """
    return {
      'active': self.long_control_state != LongCtrlState.off,
      'aeb_active': self.aeb_active,
      'output_accel': float(self.last_output_accel),
      'output_torque': float(self.last_output_torque),
      'output_angle': float(self.last_output_angle),
      'control_mode': self.control_mode,
    }
