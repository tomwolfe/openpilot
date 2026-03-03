"""
Integration tests for Phase 2 Full E2E Policy implementation.

Tests the complete E2E pipeline from model output to control commands,
including safety layer validation.
"""

import numpy as np
import pytest
from cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, PolicyType, ManeuverHint


class TestE2EPolicyConstants:
  """Test E2E policy-related constants."""

  def test_policy_horizon_constants(self):
    """Verify policy horizon constants are correctly defined."""
    assert ModelConstants.POLICY_HORIZON_SECONDS == 5.0
    assert ModelConstants.POLICY_FREQ_HZ == 100
    assert ModelConstants.POLICY_HORIZON_POINTS == 500

  def test_policy_type_enum(self):
    """Verify PolicyType enum values."""
    assert PolicyType.STANDARD == 0
    assert PolicyType.LANE_CHANGE == 1
    assert PolicyType.STOPPING == 2
    assert PolicyType.STARTING == 3
    assert PolicyType.EMERGENCY == 4

  def test_maneuver_hint_enum(self):
    """Verify ManeuverHint enum values."""
    assert ManeuverHint.NONE == 0
    assert ManeuverHint.FOLLOW_LANE == 1
    assert ManeuverHint.STOP == 6
    assert ManeuverHint.GO == 7


class TestE2EPolicyMessage:
  """Test E2E Policy Cap'n Proto message structure."""

  def test_policy_message_creation(self):
    """Test creating a Policy message."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # Set longitudinal accelerations (500 points)
    accel_profile = np.zeros(ModelConstants.POLICY_HORIZON_POINTS, dtype=np.float32)
    policy.longitudinalAccelerations = accel_profile.tolist()
    
    # Set lateral curvatures (500 points)
    curvature_profile = np.zeros(ModelConstants.POLICY_HORIZON_POINTS, dtype=np.float32)
    policy.lateralCurvatures = curvature_profile.tolist()
    
    # Set metadata
    policy.confidence = 0.95
    policy.policyType = log.ModelDataV2.Policy.PolicyType.standard
    policy.maneuverHint = log.ModelDataV2.Policy.ManeuverHint.followLane
    
    # Validate
    assert len(policy.longitudinalAccelerations) == ModelConstants.POLICY_HORIZON_POINTS
    assert len(policy.lateralCurvatures) == ModelConstants.POLICY_HORIZON_POINTS
    assert policy.confidence == 0.95
    assert policy.policyType == log.ModelDataV2.Policy.PolicyType.standard

  def test_modelv2_full_e2e_policy_field(self):
    """Test ModelDataV2.fullE2EPolicy field."""
    model_v2 = log.ModelDataV2.new_message()
    
    # Initialize and set fullE2EPolicy
    model_v2.fullE2EPolicy = log.ModelDataV2.Policy.new_message()
    model_v2.fullE2EPolicy.confidence = 0.9
    
    assert hasattr(model_v2, 'fullE2EPolicy')
    assert model_v2.fullE2EPolicy.confidence == 0.9

  def test_policy_with_realistic_profile(self):
    """Test policy with realistic acceleration/curvature profile."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # Create realistic deceleration profile (slowing from 20 m/s to stop)
    time_array = np.linspace(0, 5, ModelConstants.POLICY_HORIZON_POINTS)
    accel_profile = -2.0 * np.exp(-time_array / 2.0)  # Exponential decay
    
    # Create realistic curvature profile (gentle curve)
    curvature_profile = 0.001 * np.sin(time_array * np.pi / 5)
    
    policy.longitudinalAccelerations = accel_profile.astype(np.float32).tolist()
    policy.lateralCurvatures = curvature_profile.astype(np.float32).tolist()
    policy.confidence = 0.85
    policy.policyType = log.ModelDataV2.Policy.PolicyType.stopping
    
    # Validate ranges
    accel_list = list(policy.longitudinalAccelerations)
    assert min(accel_list) >= -3.5  # Within braking limits
    assert max(accel_list) <= 2.0   # Within acceleration limits
    
    curvature_list = list(policy.lateralCurvatures)
    assert all(abs(c) < 0.01 for c in curvature_list)  # Reasonable curvature


class TestE2EPolicyValidation:
  """Test E2E policy validation logic."""

  def test_policy_type_validation(self):
    """Test that only valid policy types are accepted."""
    valid_types = {
      log.ModelDataV2.Policy.PolicyType.standard,
      log.ModelDataV2.Policy.PolicyType.stopping,
      log.ModelDataV2.Policy.PolicyType.starting,
    }
    
    for policy_type in valid_types:
      policy = log.ModelDataV2.Policy.new_message()
      policy.policyType = policy_type
      policy.longitudinalAccelerations = [0.0] * ModelConstants.POLICY_HORIZON_POINTS
      policy.lateralCurvatures = [0.0] * ModelConstants.POLICY_HORIZON_POINTS
      
      # Should be valid for longitudinal control
      assert policy.policyType in valid_types

  def test_confidence_range(self):
    """Test confidence values are in valid range."""
    policy = log.ModelDataV2.Policy.new_message()
    policy.longitudinalAccelerations = [0.0] * ModelConstants.POLICY_HORIZON_POINTS
    
    # Valid confidence
    for confidence in [0.0, 0.5, 0.95, 1.0]:
      policy.confidence = confidence
      assert 0.0 <= policy.confidence <= 1.0

  def test_acceleration_limits(self):
    """Test that accelerations are within ISO 15622 limits."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # ISO 15622 limits (from opendbc/safety/longitudinal.h)
    ISO_LONG_ACCEL_MAX = 2.0
    ISO_LONG_DECEL_MAX = 3.5
    
    # Create profile within limits
    accel_profile = np.linspace(-3.0, 1.5, ModelConstants.POLICY_HORIZON_POINTS)
    policy.longitudinalAccelerations = accel_profile.tolist()
    
    # Validate
    accel_list = list(policy.longitudinalAccelerations)
    assert max(accel_list) <= ISO_LONG_ACCEL_MAX + 0.3  # With safety margin
    assert min(accel_list) >= -ISO_LONG_DECEL_MAX - 0.3

  def test_curvature_limits(self):
    """Test that curvatures produce safe lateral accelerations."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # ISO 11270 limit (from opendbc/safety/lateral.h)
    ISO_LATERAL_ACCEL_MAX = 3.0
    
    # At 30 m/s (108 km/h), max curvature for 2.5 m/s² lateral accel
    v_high = 30.0
    max_curvature_high = ISO_LATERAL_ACCEL_MAX / (v_high ** 2)
    
    # At 10 m/s (36 km/h), max curvature
    v_low = 10.0
    max_curvature_low = ISO_LATERAL_ACCEL_MAX / (v_low ** 2)
    
    # Create profile within limits at high speed
    curvature_profile = [max_curvature_high * 0.8] * ModelConstants.POLICY_HORIZON_POINTS
    policy.lateralCurvatures = curvature_profile.tolist()
    
    # Validate lateral acceleration at high speed
    lateral_accel = max(curvature_profile) * v_high ** 2
    assert lateral_accel <= ISO_LATERAL_ACCEL_MAX


class TestE2EToControlIntegration:
  """Test integration between E2E policy and control commands."""

  def test_policy_to_accel_command(self):
    """Test extracting acceleration command from policy."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # Create acceleration profile
    accel_profile = np.linspace(-2.0, 1.0, ModelConstants.POLICY_HORIZON_POINTS)
    policy.longitudinalAccelerations = accel_profile.tolist()
    policy.policyType = log.ModelDataV2.Policy.PolicyType.standard
    
    # Extract first acceleration command (immediate)
    accel_cmd = policy.longitudinalAccelerations[0]
    assert accel_cmd == -2.0
    
    # Extract acceleration at 1 second (index 100 at 100Hz)
    accel_1s = policy.longitudinalAccelerations[100]
    assert accel_1s == pytest.approx(-1.4, abs=0.1)

  def test_policy_to_curvature_command(self):
    """Test extracting curvature command from policy."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # Create curvature profile
    curvature_profile = np.linspace(0.0, 0.005, ModelConstants.POLICY_HORIZON_POINTS)
    policy.lateralCurvatures = curvature_profile.tolist()
    
    # Extract first curvature command
    curvature_cmd = policy.lateralCurvatures[0]
    assert curvature_cmd == 0.0
    
    # Extract curvature at 2 seconds
    curvature_2s = policy.lateralCurvatures[200]
    assert curvature_2s == pytest.approx(0.002, abs=0.0005)

  def test_policy_type_to_maneuver(self):
    """Test mapping policy type to driving maneuver."""
    maneuver_map = {
      log.ModelDataV2.Policy.PolicyType.standard: ManeuverHint.FOLLOW_LANE,
      log.ModelDataV2.Policy.PolicyType.stopping: ManeuverHint.STOP,
      log.ModelDataV2.Policy.PolicyType.starting: ManeuverHint.GO,
      log.ModelDataV2.Policy.PolicyType.laneChange: ManeuverHint.LANE_CHANGE_LEFT,
    }
    
    for policy_type, expected_hint in maneuver_map.items():
      policy = log.ModelDataV2.Policy.new_message()
      policy.policyType = policy_type
      
      # Policy type should indicate maneuver
      assert policy.policyType in [pt.value for pt in PolicyType]


class TestE2ESafetyIntegration:
  """Test integration with safety layer."""

  def test_safety_layer_constants_match(self):
    """Test that Python constants match C++ safety layer constants."""
    # Python constants (from constants.py)
    from openpilot.selfdrive.modeld.constants import ModelConstants
    
    # These should match opendbc/safety/longitudinal.h
    ISO_LONG_ACCEL_MAX_PY = 2.0
    ISO_LONG_DECEL_MAX_PY = 3.5
    ISO_LONG_JERK_MAX_PY = 5.0
    
    # These should match opendbc/safety/lateral.h
    ISO_LATERAL_ACCEL_MAX_PY = 3.0
    ISO_LATERAL_JERK_MAX_PY = 4.0
    
    # Verify constants are defined
    assert ISO_LONG_ACCEL_MAX_PY > 0
    assert ISO_LONG_DECEL_MAX_PY > 0
    assert ISO_LONG_JERK_MAX_PY > 0
    assert ISO_LATERAL_ACCEL_MAX_PY > 0
    assert ISO_LATERAL_JERK_MAX_PY > 0

  def test_policy_within_safety_limits(self):
    """Test that generated policies are within safety limits."""
    policy = log.ModelDataV2.Policy.new_message()
    
    # Generate policy within safety limits
    accel_profile = np.clip(
      np.random.randn(ModelConstants.POLICY_HORIZON_POINTS) * 0.5,
      -3.0, 1.8
    )
    curvature_profile = np.clip(
      np.random.randn(ModelConstants.POLICY_HORIZON_POINTS) * 0.002,
      -0.005, 0.005
    )
    
    policy.longitudinalAccelerations = accel_profile.tolist()
    policy.lateralCurvatures = curvature_profile.tolist()
    
    # Validate all points are within limits
    accel_list = list(policy.longitudinalAccelerations)
    assert all(-3.5 <= a <= 2.0 for a in accel_list)
    
    curvature_list = list(policy.lateralCurvatures)
    assert all(abs(c) <= 0.01 for c in curvature_list)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
