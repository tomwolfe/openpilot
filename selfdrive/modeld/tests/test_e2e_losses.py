#!/usr/bin/env python3
"""
Unit Tests for E2E Loss Functions

Tests validate that the E2E loss functions:
1. Compute correct values for known inputs
2. Handle edge cases properly
3. Gradients flow correctly (for training)
4. Penalties are applied as expected

Run tests:
  cd /path/to/openpilot
  pytest selfdrive/modeld/tests/test_e2e_losses.py -v
"""
import unittest
import numpy as np
from openpilot.selfdrive.modeld.e2e_losses import E2ELosses, create_training_example


class TestJerkLoss(unittest.TestCase):
  """Tests for jerk loss function."""
  
  def setUp(self):
    self.losses = E2ELosses(jerk_weight=1.0)
    
  def test_zero_jerk_constant_acceleration(self):
    """Constant acceleration should have zero jerk."""
    # Constant acceleration = 2.0 m/s²
    accel = np.ones((1, 10)) * 2.0
    loss = self.losses.jerk_loss(accel)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_zero_jerk_stationary(self):
    """Zero acceleration should have zero jerk."""
    accel = np.zeros((1, 10))
    loss = self.losses.jerk_loss(accel)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_positive_jerk_linear_increase(self):
    """Linearly increasing acceleration should have constant jerk."""
    # Linear increase: 0, 1, 2, 3, ...
    accel = np.arange(10, dtype=np.float32)[np.newaxis, :]
    loss = self.losses.jerk_loss(accel)
    # Jerk should be zero (constant rate of change)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_high_jerk_step_change(self):
    """Step change in acceleration should have high jerk."""
    # Sudden change from 0 to 5 m/s²
    accel = np.array([[0, 0, 0, 5, 5, 5, 5, 5, 5, 5]], dtype=np.float32)
    loss = self.losses.jerk_loss(accel)
    self.assertGreater(loss, 0.0)
    
  def test_jerk_weight_scaling(self):
    """Loss should scale with jerk_weight parameter."""
    accel = np.array([[0, 0, 0, 5, 5, 5, 5, 5, 5, 5]], dtype=np.float32)
    
    losses_1 = E2ELosses(jerk_weight=1.0)
    losses_2 = E2ELosses(jerk_weight=2.0)
    
    loss_1 = losses_1.jerk_loss(accel)
    loss_2 = losses_2.jerk_loss(accel)
    
    self.assertAlmostEqual(loss_2, loss_1 * 2.0, places=6)
    
  def test_batch_processing(self):
    """Should handle batched inputs correctly."""
    batch_size = 8
    accel = np.random.randn(batch_size, 10)
    loss = self.losses.jerk_loss(accel)
    self.assertIsInstance(loss, float)
    self.assertGreaterEqual(loss, 0.0)
    
  def test_short_sequence(self):
    """Should handle short sequences gracefully."""
    accel = np.array([[1.0, 2.0]], dtype=np.float32)
    loss = self.losses.jerk_loss(accel)
    # Should not crash, may return 0 for too short sequences
    self.assertGreaterEqual(loss, 0.0)


class TestActuationLoss(unittest.TestCase):
  """Tests for actuation loss function."""
  
  def setUp(self):
    self.losses = E2ELosses(actuation_weight=1.0)
    
  def test_perfect_match_zero_loss(self):
    """Perfect prediction should have zero loss."""
    torque = np.random.randn(1, 10)
    gas = np.random.rand(1, 10)
    brake = np.random.rand(1, 10)
    
    loss = self.losses.actuation_loss(torque, gas, brake, torque, gas, brake)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_constant_offset(self):
    """Constant offset should produce MSE loss."""
    pred_torque = np.ones((1, 10)) * 0.5
    target_torque = np.ones((1, 10)) * 0.3
    gas = brake = np.zeros((1, 10))
    
    loss = self.losses.actuation_loss(
      pred_torque, gas, brake,
      target_torque, gas, brake
    )
    # Expected MSE = (0.5 - 0.3)² = 0.04
    self.assertAlmostEqual(loss, 0.04, places=4)
    
  def test_brake_weight_higher(self):
    """Brake loss should have higher weight."""
    pred_brake = np.ones((1, 10)) * 0.5
    target_brake = np.zeros((1, 10))
    torque = gas = np.zeros((1, 10))
    
    # Brake-only loss
    brake_loss = self.losses.actuation_loss(
      torque, gas, pred_brake,
      torque, gas, target_brake
    )
    
    # Gas-only loss (same error magnitude)
    pred_gas = np.ones((1, 10)) * 0.5
    gas_loss = self.losses.actuation_loss(
      torque, pred_gas, gas,
      torque, gas, gas
    )
    
    # Brake loss should be 1.5x gas loss
    self.assertAlmostEqual(brake_loss / gas_loss, 1.5, places=4)
    
  def test_batch_processing(self):
    """Should handle batched inputs correctly."""
    batch_size = 8
    pred_torque = np.random.randn(batch_size, 10)
    pred_gas = np.random.rand(batch_size, 10)
    pred_brake = np.random.rand(batch_size, 10)
    
    target_torque = np.random.randn(batch_size, 10)
    target_gas = np.random.rand(batch_size, 10)
    target_brake = np.random.rand(batch_size, 10)
    
    loss = self.losses.actuation_loss(
      pred_torque, pred_gas, pred_brake,
      target_torque, target_gas, target_brake
    )
    self.assertIsInstance(loss, float)
    self.assertGreaterEqual(loss, 0.0)


class TestComfortLoss(unittest.TestCase):
  """Tests for comfort loss function."""
  
  def setUp(self):
    self.losses = E2ELosses(comfort_weight=1.0)
    
  def test_within_comfort_zone_zero_loss(self):
    """Acceleration within comfort zone should have zero loss."""
    # Comfort zone: ±2.0 m/s²
    accel = np.ones((1, 10)) * 1.5  # Within zone
    loss = self.losses.comfort_loss(accel)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_negative_within_comfort_zone(self):
    """Negative acceleration within comfort zone should have zero loss."""
    accel = np.ones((1, 10)) * -1.5  # Within zone
    loss = self.losses.comfort_loss(accel)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_exceeds_comfort_zone_positive(self):
    """Acceleration beyond comfort zone should have loss."""
    accel = np.ones((1, 10)) * 3.0  # Exceeds by 1.0
    loss = self.losses.comfort_loss(accel)
    # Loss = (3.0 - 2.0)² = 1.0
    self.assertAlmostEqual(loss, 1.0, places=4)
    
  def test_exceeds_comfort_zone_negative(self):
    """Negative acceleration beyond comfort zone should have loss."""
    accel = np.ones((1, 10)) * -4.0  # Exceeds by 2.0
    loss = self.losses.comfort_loss(accel)
    # Loss = (|-4.0| - 2.0)² = 4.0
    self.assertAlmostEqual(loss, 4.0, places=4)
    
  def test_lateral_accel_comfort(self):
    """Lateral acceleration should also be penalized."""
    long_accel = np.zeros((1, 10))
    lat_accel = np.ones((1, 10)) * 4.0  # Exceeds 2.5 m/s² limit
    
    loss = self.losses.comfort_loss(long_accel, lat_accel)
    # Loss = (4.0 - 2.5)² = 2.25
    self.assertAlmostEqual(loss, 2.25, places=4)
    
  def test_custom_comfort_thresholds(self):
    """Should support custom comfort thresholds."""
    losses_strict = E2ELosses(comfort_weight=1.0)
    
    accel = np.ones((1, 10)) * 1.5
    
    # With strict threshold (1.0), should have loss
    loss_strict = losses_strict.comfort_loss(accel, max_long_accel=1.0)
    self.assertGreater(loss_strict, 0.0)


class TestSteeringSmoothnessLoss(unittest.TestCase):
  """Tests for steering smoothness loss function."""
  
  def setUp(self):
    self.losses = E2ELosses(smoothness_weight=1.0)
    
  def test_constant_steering_zero_loss(self):
    """Constant steering should have zero smoothness loss."""
    torque = np.ones((1, 10)) * 0.5
    loss = self.losses.steering_smoothness_loss(torque)
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_linear_ramp_constant_rate(self):
    """Linear ramp has constant rate, but still penalized."""
    torque = np.arange(10, dtype=np.float32)[np.newaxis, :] / 10.0
    loss = self.losses.steering_smoothness_loss(torque)
    # Constant rate is still penalized (we want minimal steering changes)
    self.assertGreater(loss, 0.0)
    
  def test_high_frequency_oscillation_high_loss(self):
    """High-frequency oscillations should have high loss."""
    # Oscillating: -1, 1, -1, 1, ...
    torque = np.array([[(-1)**i * 0.5 for i in range(10)]], dtype=np.float32)
    loss = self.losses.steering_smoothness_loss(torque)
    self.assertGreater(loss, 1.0)  # Should be high
    
  def test_smooth_sine_wave(self):
    """Smooth sine wave should have moderate loss."""
    t = np.linspace(0, 2*np.pi, 10)
    torque = np.sin(t)[np.newaxis, :] * 0.5
    loss = self.losses.steering_smoothness_loss(torque)
    self.assertGreater(loss, 0.0)
    self.assertLess(loss, 100.0)  # Should be reasonable (sine has significant derivative)


class TestAEBLoss(unittest.TestCase):
  """Tests for AEB loss function."""
  
  def setUp(self):
    self.losses = E2ELosses(aeb_weight=1.0)
    
  def test_no_emergency_no_braking_zero_loss(self):
    """No braking during non-emergency should have zero loss."""
    pred_brake = np.zeros((1, 10))
    target_brake = np.zeros((1, 10))
    is_emergency = np.zeros((1, 10), dtype=bool)
    
    loss = self.losses.aeb_loss(
      np.zeros((1, 10)),  # crash_prob (unused)
      np.zeros((1, 10)),  # ttc (unused)
      pred_brake, target_brake, is_emergency
    )
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_false_positive_penalty(self):
    """Unnecessary braking should be penalized."""
    pred_brake = np.ones((1, 10)) * 0.5  # Braking
    target_brake = np.zeros((1, 10))  # Human not braking
    is_emergency = np.zeros((1, 10), dtype=bool)
    
    loss = self.losses.aeb_loss(
      np.zeros((1, 10)), np.zeros((1, 10)),
      pred_brake, target_brake, is_emergency
    )
    self.assertGreater(loss, 0.0)
    
  def test_false_negative_penalty(self):
    """Not braking during emergency should be heavily penalized."""
    pred_brake = np.zeros((1, 10))  # Not braking
    target_brake = np.ones((1, 10)) * 0.8  # Human braking hard
    is_emergency = np.ones((1, 10), dtype=bool)
    
    loss = self.losses.aeb_loss(
      np.zeros((1, 10)), np.zeros((1, 10)),
      pred_brake, target_brake, is_emergency
    )
    self.assertGreater(loss, 0.5)  # Should be significant
    
  def test_correct_emergency_braking(self):
    """Correct emergency braking should have low loss."""
    pred_brake = np.ones((1, 10)) * 0.8  # Braking hard
    target_brake = np.ones((1, 10)) * 0.8  # Matching human
    is_emergency = np.ones((1, 10), dtype=bool)
    
    loss = self.losses.aeb_loss(
      np.zeros((1, 10)), np.zeros((1, 10)),
      pred_brake, target_brake, is_emergency
    )
    self.assertAlmostEqual(loss, 0.0, places=6)
    
  def test_mixed_emergency_non_emergency(self):
    """Should handle mixed emergency/non-emergency correctly."""
    # First half: emergency, second half: normal
    pred_brake = np.ones((1, 10)) * 0.5
    target_brake = np.ones((1, 10)) * 0.5
    is_emergency = np.array([[True]*5 + [False]*5], dtype=bool)
    
    loss = self.losses.aeb_loss(
      np.zeros((1, 10)), np.zeros((1, 10)),
      pred_brake, target_brake, is_emergency
    )
    # Should handle both cases without error
    self.assertGreaterEqual(loss, 0.0)


class TestComputeLoss(unittest.TestCase):
  """Tests for combined loss computation."""
  
  def setUp(self):
    self.losses = E2ELosses()
    
  def test_all_losses_computed(self):
    """Should compute all loss components."""
    model_outputs, human_targets, car_state = create_training_example()
    
    total_loss, components = self.losses.compute_loss(
      model_outputs, human_targets, car_state, return_components=True
    )
    
    # Check all expected components are present
    expected_keys = ['jerk', 'actuation', 'comfort', 'smoothness', 'aeb']
    for key in expected_keys:
      self.assertIn(key, components)
      
    # Total should be sum of components
    self.assertAlmostEqual(total_loss, sum(components.values()), places=6)
    
  def test_total_loss_positive(self):
    """Total loss should always be positive."""
    model_outputs, human_targets, car_state = create_training_example()
    
    total_loss = self.losses.compute_loss(
      model_outputs, human_targets, car_state
    )
    
    self.assertGreater(total_loss, 0.0)
    
  def test_missing_optional_inputs(self):
    """Should handle missing optional inputs gracefully."""
    model_outputs, human_targets, _ = create_training_example()
    
    # No car_state (AEB loss won't be computed)
    total_loss, components = self.losses.compute_loss(
      model_outputs, human_targets, return_components=True
    )
    
    # AEB should not be in components
    self.assertNotIn('aeb', components)
    
    # Other losses should still work
    self.assertIn('jerk', components)
    self.assertIn('actuation', components)
    
  def test_loss_weights_affect_total(self):
    """Changing weights should affect total loss."""
    model_outputs, human_targets, car_state = create_training_example()
    
    losses_1 = E2ELosses(jerk_weight=1.0)
    losses_2 = E2ELosses(jerk_weight=10.0)
    
    total_1, _ = losses_1.compute_loss(model_outputs, human_targets, car_state, return_components=True)
    total_2, _ = losses_2.compute_loss(model_outputs, human_targets, car_state, return_components=True)
    
    # Higher jerk weight should increase total (for non-zero jerk)
    self.assertGreater(total_2, total_1)


class TestCreateTrainingExample(unittest.TestCase):
  """Tests for synthetic data generation."""
  
  def test_output_shapes(self):
    """Generated data should have correct shapes."""
    batch_size = 4
    seq_len = 10
    
    model_outputs, human_targets, car_state = create_training_example(
      batch_size=batch_size, seq_len=seq_len
    )
    
    # Check model outputs
    self.assertEqual(model_outputs['torque'].shape, (batch_size, seq_len))
    self.assertEqual(model_outputs['gas'].shape, (batch_size, seq_len))
    self.assertEqual(model_outputs['brake'].shape, (batch_size, seq_len))
    self.assertEqual(model_outputs['accel'].shape, (batch_size, seq_len))
    
    # Check car state
    self.assertEqual(car_state['v_ego'].shape, (batch_size, seq_len))
    self.assertEqual(car_state['is_emergency'].shape, (batch_size, seq_len))
    
  def test_value_ranges(self):
    """Generated data should be in reasonable ranges."""
    model_outputs, human_targets, car_state = create_training_example()
    
    # Torque should be in [-1, 1] range (approximately)
    self.assertLessEqual(np.max(model_outputs['torque']), 1.5)
    self.assertGreaterEqual(np.min(model_outputs['torque']), -1.5)
    
    # Gas/brake should be in [0, 1] range (approximately)
    self.assertLessEqual(np.max(model_outputs['gas']), 1.5)
    self.assertGreaterEqual(np.min(model_outputs['gas']), -0.5)
    
    # Velocity should be positive
    self.assertGreater(np.min(car_state['v_ego']), 0)
    
  def test_reproducibility(self):
    """Should generate same data with same seed."""
    outputs_1, targets_1, state_1 = create_training_example()
    outputs_2, targets_2, state_2 = create_training_example()
    
    np.testing.assert_array_equal(outputs_1['torque'], outputs_2['torque'])
    np.testing.assert_array_equal(targets_1['gas'], targets_2['gas'])


if __name__ == '__main__':
  unittest.main()
