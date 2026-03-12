#!/usr/bin/env python3
"""
E2E Phase 2: Direct Longitudinal Control

This controller bypasses the classical PID controller and directly applies
the model's predicted gas/brake commands to the actuators.

The neural network outputs gas_pred and brake_pred which are mapped directly
to CC.actuators.accel after simple low-pass filtering.
"""
import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants

LongCtrlState = car.CarControl.Actuators.LongControlState


class LongControlE2E:
  """
  E2E Phase 2: Direct actuation longitudinal controller.
  
  Instead of calculating tracking error against a_target and using a PID controller,
  this directly applies the model's gas/brake predictions with minimal filtering.
  """
  
  def __init__(self, CP):
    self.CP = CP
    
    # Low-pass filter for smoothing actuator commands
    # Time constant tuned for smooth but responsive control
    LONG_FILTER_SECONDS = 0.1
    self.gas_filter = FirstOrderFilter(0.0, LONG_FILTER_SECONDS, DT_CTRL)
    self.brake_filter = FirstOrderFilter(0.0, LONG_FILTER_SECONDS, DT_CTRL)
    
    self.long_control_state = LongCtrlState.off
    self.last_output_accel = 0.0
    
    # AEB override state
    self.aeb_active = False
    
  def reset(self):
    """Reset controller state and filters."""
    self.gas_filter.x = 0.0
    self.brake_filter.x = 0.0
    self.last_output_accel = 0.0
    self.long_control_state = LongCtrlState.off
    self.aeb_active = False
    
  def update(self, active, CS, model_output, accel_limits, aeb_override=None):
    """
    Update longitudinal control using direct E2E actuation.
    
    Args:
      active: Whether longitudinal control is active
      CS: CarState message
      model_output: Dict containing model predictions including 'gas_pred' and 'brake_pred'
      accel_limits: [min_accel, max_accel] limits from vehicle
      aeb_override: Optional AEB acceleration override (when model detects imminent collision)
    
    Returns:
      output_accel: Acceleration command to send to actuators
    """
    # Extract model predictions
    # Model outputs are in range [0, 1] representing pedal position
    gas_pred = model_output.get('gas_pred', 0.0)
    brake_pred = model_output.get('brake_pred', 0.0)
    
    # Apply low-pass filtering for smooth actuation
    filtered_gas = self.gas_filter.update(gas_pred)
    filtered_brake = self.brake_filter.update(brake_pred)
    
    # Convert pedal positions to acceleration
    # Gas: 0-1 maps to [0, max_accel]
    # Brake: 0-1 maps to [min_accel, 0]
    gas_accel = filtered_gas * accel_limits[1]
    brake_accel = filtered_brake * accel_limits[0]
    
    # Combine gas and brake (they should be mutually exclusive in a well-trained model)
    # Net acceleration is gas - brake effect
    output_accel = gas_accel + brake_accel
    
    # E2E AEB override: if model predicts imminent collision, override with maximum braking
    if aeb_override is not None and aeb_override < output_accel:
      output_accel = aeb_override
      self.aeb_active = True
    else:
      self.aeb_active = False
    
    # Clip to vehicle limits
    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    
    if not active:
      self.reset()
      self.last_output_accel = 0.0
      self.long_control_state = LongCtrlState.off
    else:
      self.long_control_state = LongCtrlState.pid  # Reuse pid state for compatibility
    
    return self.last_output_accel
  
  def get_aeb_status(self):
    """Returns whether AEB is currently active."""
    return self.aeb_active
