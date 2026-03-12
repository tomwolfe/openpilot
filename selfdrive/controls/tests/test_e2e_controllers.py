#!/usr/bin/env python3
"""
Unit Tests for E2E Controllers

Tests validate that the E2E controllers:
1. Process model outputs correctly
2. Apply appropriate filtering
3. Handle edge cases (AEB, disengagement)
4. Produce bounded outputs

Run tests:
  cd /path/to/openpilot
  pytest selfdrive/controls/tests/test_e2e_controllers.py -v
"""
import unittest
import numpy as np
from unittest.mock import Mock


class TestLongControlE2E(unittest.TestCase):
  """Tests for E2E longitudinal controller."""
  
  def setUp(self):
    """Set up test fixtures."""
    # Mock CarParams
    self.CP = Mock()
    self.CP.minSteerSpeed = 0.3
    self.CP.steerLimitTimer = 1.0
    
    # Import controller (deferred to avoid cereal import issues)
    from openpilot.selfdrive.controls.lib.longcontrol import LongControl
    self.LongControl = LongControl
    
  def test_controller_initialization(self):
    """Controller should initialize without errors."""
    controller = self.LongControl(self.CP)
    self.assertIsNotNone(controller)
    self.assertEqual(controller.long_control_state, 0)  # off state
    
  def test_zero_model_output_zero_accel(self):
    """Zero model outputs should produce zero acceleration."""
    controller = self.LongControl(self.CP)
    
    # Mock CarState
    CS = Mock()
    CS.vEgo = 20.0
    CS.brakePressed = False
    CS.gasPressed = False
    
    model_output = {'gas_pred': 0.0, 'brake_pred': 0.0}
    accel_limits = [-4.0, 2.0]
    
    accel = controller.update(True, CS, model_output, accel_limits)
    self.assertAlmostEqual(accel, 0.0, places=4)
    
  def test_gas_prediction_acceleration(self):
    """Gas prediction should produce positive acceleration."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    CS.brakePressed = False
    CS.gasPressed = False
    
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    accel_limits = [-4.0, 2.0]
    
    accel = controller.update(True, CS, model_output, accel_limits)
    self.assertGreater(accel, 0.0)
    self.assertLessEqual(accel, accel_limits[1])
    
  def test_brake_prediction_deceleration(self):
    """Brake prediction should produce negative acceleration."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    CS.brakePressed = False
    CS.gasPressed = False
    
    model_output = {'gas_pred': 0.0, 'brake_pred': 0.5}
    accel_limits = [-4.0, 2.0]
    
    accel = controller.update(True, CS, model_output, accel_limits)
    self.assertLess(accel, 0.0)
    self.assertGreaterEqual(accel, accel_limits[0])
    
  def test_aeb_override(self):
    """AEB override should apply maximum braking."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    CS.brakePressed = False
    CS.gasPressed = False
    
    # Model wants light gas, but AEB overrides
    model_output = {'gas_pred': 0.2, 'brake_pred': 0.0}
    accel_limits = [-4.0, 2.0]
    aeb_override = -4.0  # Maximum braking
    
    accel = controller.update(True, CS, model_output, accel_limits, aeb_override)
    self.assertAlmostEqual(accel, -4.0, places=4)
    
  def test_aeb_status_flag(self):
    """AEB status flag should be set correctly."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    # No AEB
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    controller.update(True, CS, model_output, [-4.0, 2.0])
    self.assertFalse(controller.get_aeb_status())
    
    # AEB active
    controller.update(True, CS, model_output, [-4.0, 2.0], aeb_override=-4.0)
    self.assertTrue(controller.get_aeb_status())
    
  def test_acceleration_clipping(self):
    """Acceleration should be clipped to vehicle limits."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    # Model predicts beyond limits
    model_output = {'gas_pred': 1.5, 'brake_pred': 0.0}  # Beyond [0, 1]
    accel_limits = [-4.0, 2.0]
    
    accel = controller.update(True, CS, model_output, accel_limits)
    self.assertLessEqual(accel, accel_limits[1])
    self.assertGreaterEqual(accel, accel_limits[0])
    
  def test_filter_smoothing(self):
    """Low-pass filter should smooth step changes."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    # Step change from 0 to 0.5
    model_output_step = {'gas_pred': 0.5, 'brake_pred': 0.0}
    model_output_zero = {'gas_pred': 0.0, 'brake_pred': 0.0}
    
    accel_limits = [-4.0, 2.0]
    
    # First update (step)
    accel1 = controller.update(True, CS, model_output_step, accel_limits)
    
    # Second update (back to zero) - should be filtered
    accel2 = controller.update(True, CS, model_output_zero, accel_limits)
    
    # accel2 should be between accel1 and 0 (filtered)
    self.assertLess(abs(accel2), abs(accel1))
    
  def test_disengagement_reset(self):
    """Controller should reset on disengagement."""
    controller = self.LongControl(self.CP)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    accel_limits = [-4.0, 2.0]
    
    # Engage
    controller.update(True, CS, model_output, accel_limits)
    self.assertNotEqual(controller.long_control_state, 0)
    
    # Disengage
    controller.update(False, CS, model_output, accel_limits)
    self.assertEqual(controller.long_control_state, 0)  # off state


class TestLatControlE2E(unittest.TestCase):
  """Tests for E2E lateral controller."""
  
  def setUp(self):
    """Set up test fixtures."""
    # Mock CarParams
    self.CP = Mock()
    self.CP.minSteerSpeed = 0.3
    self.CP.steerLimitTimer = 1.0
    self.CP.steerControlType = 0  # torque
    
    # Mock CarInterface
    self.CI = Mock()
    
    from openpilot.selfdrive.controls.lib.latcontrol import LatControl
    self.LatControl = LatControl
    
  def test_controller_initialization(self):
    """Controller should initialize without errors."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    self.assertIsNotNone(controller)
    
  def test_zero_model_output_zero_torque(self):
    """Zero model outputs should produce zero torque."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    CS.steeringTorque = 0.0
    CS.steeringTorqueEps = 0.0
    
    model_output = {'steer_torque_pred': 0.0, 'steer_angle_pred': 0.0}
    
    torque, angle, log = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertAlmostEqual(torque, 0.0, places=4)
    
  def test_torque_prediction(self):
    """Torque prediction should be passed through."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    CS.steeringTorque = 0.0
    CS.steeringTorqueEps = 0.0
    
    model_output = {'steer_torque_pred': 0.5, 'steer_angle_pred': 0.0}
    
    torque, angle, log = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertAlmostEqual(torque, 0.5, places=4)
    
  def test_torque_clipping(self):
    """Torque should be clipped to [-1, 1]."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {'steer_torque_pred': 2.0, 'steer_angle_pred': 0.0}  # Beyond [-1, 1]
    
    torque, angle, log = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertLessEqual(abs(torque), 1.0)
    
  def test_angle_conversion(self):
    """Angle should be converted from radians to degrees."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    import math
    model_output = {'steer_torque_pred': 0.0, 'steer_angle_pred': math.pi/4}  # 45 degrees
    
    torque, angle, log = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertAlmostEqual(angle, 45.0, places=4)
    
  def test_disengagement_reset(self):
    """Controller should reset on disengagement."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {'steer_torque_pred': 0.5, 'steer_angle_pred': 0.0}
    
    # Engage
    torque1, angle1, log1 = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertTrue(log1.active)
    
    # Disengage
    torque2, angle2, log2 = controller.update(False, CS, None, None, False, 0.0, False, 0.1, model_output)
    self.assertFalse(log2.active)
    self.assertAlmostEqual(torque2, 0.0, places=4)
    
  def test_filter_smoothing(self):
    """Low-pass filter should smooth step changes."""
    controller = self.LatControl(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output_step = {'steer_torque_pred': 0.8, 'steer_angle_pred': 0.0}
    model_output_zero = {'steer_torque_pred': 0.0, 'steer_angle_pred': 0.0}
    
    # First update (step)
    torque1, _, _ = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output_step)
    
    # Second update (back to zero) - should be filtered
    torque2, _, _ = controller.update(True, CS, None, None, False, 0.0, False, 0.1, model_output_zero)
    
    # torque2 should be between torque1 and 0 (filtered)
    self.assertLess(abs(torque2), abs(torque1))


class TestE2EControllerUnified(unittest.TestCase):
  """Tests for unified E2E controller."""
  
  def setUp(self):
    """Set up test fixtures."""
    from openpilot.selfdrive.controls.lib.e2e_controller import E2EController
    
    # Mock CarParams
    self.CP = Mock()
    self.CP.minSteerSpeed = 0.3
    self.CP.steerLimitTimer = 1.0
    self.CP.steerControlType = 0  # torque
    
    # Mock CarInterface
    self.CI = Mock()
    
    self.E2EController = E2EController
    
  def test_unified_controller_init(self):
    """Unified controller should initialize correctly."""
    controller = self.E2EController(self.CP, self.CI, 0.01)
    self.assertIsNotNone(controller)
    self.assertIn(controller.control_mode, ['torque', 'angle'])
    
  def test_unified_control_outputs(self):
    """Should produce all actuator outputs."""
    controller = self.E2EController(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {
      'gas_pred': 0.3,
      'brake_pred': 0.0,
      'steer_torque_pred': 0.2,
      'steer_angle_pred': 0.0,
      'crash_prob': 0.0,
      'ttc_pred': 5.0,
    }
    
    accel_limits = [-4.0, 2.0]
    
    accel, torque, angle = controller.update(True, CS, model_output, accel_limits)
    
    # All outputs should be valid
    self.assertIsInstance(accel, float)
    self.assertIsInstance(torque, float)
    self.assertIsInstance(angle, float)
    
  def test_unified_aeb_affects_longitudinal_only(self):
    """AEB should only affect longitudinal control."""
    controller = self.E2EController(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {
      'gas_pred': 0.5,
      'brake_pred': 0.0,
      'steer_torque_pred': 0.2,
      'steer_angle_pred': 0.0,
      'crash_prob': 0.0,
      'ttc_pred': 5.0,
    }
    
    accel_limits = [-4.0, 2.0]
    
    # Normal operation
    accel1, torque1, _ = controller.update(True, CS, model_output, accel_limits)
    
    # AEB operation
    accel2, torque2, _ = controller.update(True, CS, model_output, accel_limits, aeb_override=-4.0)
    
    # Longitudinal should change
    self.assertLess(accel2, accel1)
    
    # Lateral should be unchanged
    self.assertAlmostEqual(torque1, torque2, places=4)
    
  def test_controller_state_logging(self):
    """Should produce valid state for logging."""
    controller = self.E2EController(self.CP, self.CI, 0.01)
    
    CS = Mock()
    CS.vEgo = 20.0
    
    model_output = {
      'gas_pred': 0.3,
      'brake_pred': 0.0,
      'steer_torque_pred': 0.2,
      'steer_angle_pred': 0.0,
      'crash_prob': 0.0,
      'ttc_pred': 5.0,
    }
    
    controller.update(True, CS, model_output, [-4.0, 2.0])
    
    state = controller.get_controller_state()
    
    # State should have all expected fields
    self.assertIn('active', state)
    self.assertIn('aeb_active', state)
    self.assertIn('output_accel', state)
    self.assertIn('output_torque', state)
    self.assertIn('output_angle', state)
    self.assertIn('control_mode', state)


if __name__ == '__main__':
  unittest.main()
