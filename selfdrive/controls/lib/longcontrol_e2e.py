#!/usr/bin/env python3
"""
E2E Phase 2: Direct Longitudinal Control

This controller bypasses the classical PID controller and directly applies
the model's predicted gas/brake commands to the actuators.

The neural network outputs gas_pred and brake_pred which are mapped directly
to CC.actuators.accel after simple low-pass filtering.

E2E Architecture:
- No state machine (model learns when to stop/starting from data)
- No heuristic smoothing (model outputs are already smooth from training)
- No artificial limits (Panda safety layer enforces absolute bounds)
- Direct brake passthrough at all speeds including standstill
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
  
  Key differences from classical LongControl:
  - No state machine (off/pid/stopping/starting) - model learns driving states
  - No heuristic stopping logic (vEgoStopping removed)
  - Direct brake passthrough at low speeds for smooth stops
  - Model outputs holding brake pressure when stopped
  """

  def __init__(self, CP):
    self.CP = CP

    # Low-pass filter for smoothing actuator commands
    # E2E Phase 2: Reduced filter time constant - model outputs are already smooth
    # from training on human driving data. Minimal filtering preserves model's
    # intended trajectory while removing high-frequency noise.
    LONG_FILTER_SECONDS = 0.08  # Reduced from 0.1 for more direct control
    self.gas_filter = FirstOrderFilter(0.0, LONG_FILTER_SECONDS, DT_CTRL)
    self.brake_filter = FirstOrderFilter(0.0, LONG_FILTER_SECONDS, DT_CTRL)

    self.long_control_state = LongCtrlState.off
    self.last_output_accel = 0.0

    # AEB override state
    self.aeb_active = False
    
    # Low speed threshold for direct brake passthrough
    # E2E Phase 2: Model controls braking at all speeds - no artificial override
    self.low_speed_threshold = 0.5  # m/s

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

    E2E Phase 2: Direct brake passthrough at all speeds.
    The model learns appropriate holding brake pressure for stops from human data.
    No artificial smoothing or override at low speeds.

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
    # E2E Phase 2: Filter preserves model's intended trajectory
    filtered_gas = self.gas_filter.update(gas_pred)
    filtered_brake = self.brake_filter.update(brake_pred)

    # E2E Phase 2: Direct brake passthrough at all speeds
    # Classical controllers would smooth or override brake commands at low speeds,
    # but the E2E model has learned appropriate stopping behavior from human data.
    # The model outputs the correct holding brake pressure when stopped.
    
    # Convert pedal positions to acceleration
    # Gas: 0-1 maps to [0, max_accel]
    # Brake: 0-1 maps to [min_accel, 0]
    # E2E Phase 2: No artificial blending - model learns smooth transitions
    gas_accel = filtered_gas * accel_limits[1]
    brake_accel = filtered_brake * accel_limits[0]

    # Combine gas and brake (they should be mutually exclusive in a well-trained model)
    # Net acceleration is gas - brake effect
    output_accel = gas_accel + brake_accel

    # E2E AEB override: if model predicts imminent collision, override with maximum braking
    # This bypasses all comfort limits for safety
    if aeb_override is not None and aeb_override < output_accel:
      output_accel = aeb_override
      self.aeb_active = True
    else:
      self.aeb_active = False

    # Clip to vehicle limits (Panda safety layer enforces absolute bounds)
    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])

    if not active:
      self.reset()
      self.last_output_accel = 0.0
      self.long_control_state = LongCtrlState.off
    else:
      # E2E Phase 2: Simplified state - model handles all driving states
      # No separate stopping/starting states - model learns these behaviors
      self.long_control_state = LongCtrlState.pid

    return self.last_output_accel

  def get_aeb_status(self):
    """Returns whether AEB is currently active."""
    return self.aeb_active
