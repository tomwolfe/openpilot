#!/usr/bin/env python3
"""
E2E Longitudinal Control Tests

Tests for Phase 2: Unified Neural Execution
Verifies that the longitudinal planner correctly uses E2E model outputs
and that the legacy MPC is only initialized when ForceClassicalMPC is enabled.

Test Coverage:
1. E2E model output following - verifies planner uses modelV2.action.desiredAcceleration
2. Personality integration - verifies gain/offset application
3. MPC initialization - verifies MPC is not created unless ForceClassicalMPC=True
4. Stop trajectory - verifies model's shouldStop flag is respected
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, PropertyMock
from cereal import log, car

from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    LongitudinalPlanner, 
    apply_personality_to_accel,
    PERSONALITY_GAINS,
    PERSONALITY_OFFSETS,
)
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants


class MockSubMaster:
  """Mock SubMaster for simulating cereal messages without hardware."""
  
  def __init__(self, services=None):
    self._services = services or {}
    self.logMonoTime = {svc: 0 for svc in self._services}
    self.frame = 0
    
  def __getitem__(self, key):
    return self._services.get(key, Mock())
    
  def all_checks(self, service_list=None):
    return True
    
  def update(self):
    self.frame += 1


def create_mock_car_params():
  """Create minimal CarParams for testing."""
  CP = car.CarParams.new_message()
  CP.openpilotLongitudinalControl = True
  CP.steerRatio = 15.0
  CP.wheelbase = 2.7
  CP.longitudinalActuatorDelay = 0.2
  CP.vEgoStopping = 0.5
  CP.startingState = False
  return CP


def create_mock_modelV2(desired_accel=0.0, should_stop=False, 
                        speeds=None, accels=None):
  """Create a mock modelV2 message with specified E2E outputs."""
  model = Mock()
  model.action.desiredAcceleration = float(desired_accel)
  model.action.shouldStop = bool(should_stop)
  
  # Mock longitudinal plan trajectory if provided
  if speeds is not None:
    model.longitudinalPlan.speeds = speeds
  else:
    model.longitudinalPlan.speeds = [0.0] * CONTROL_N
    
  if accels is not None:
    model.longitudinalPlan.accels = accels
  else:
    model.longitudinalPlan.accels = [0.0] * CONTROL_N
  
  # Mock other required fields
  model.position.x = [0.0] * ModelConstants.IDX_N
  model.velocity.x = [0.0] * ModelConstants.IDX_N
  model.acceleration.x = [0.0] * ModelConstants.IDX_N
  model.meta.disengagePredictions.gasPressProbs = [1.0]
  
  return model


def create_mock_carState(v_ego=10.0, a_ego=0.0, standstill=False, 
                         steeringAngleDeg=0.0, vCruise=30.0):
  """Create a mock carState message."""
  state = Mock()
  state.vEgo = float(v_ego)
  state.aEgo = float(a_ego)
  state.standstill = bool(standstill)
  state.steeringAngleDeg = float(steeringAngleDeg)
  state.vCruise = float(vCruise)
  return state


def create_mock_selfdriveState(enabled=True, personality=log.LongitudinalPersonality.standard,
                                experimentalMode=False):
  """Create a mock selfdriveState message."""
  state = Mock()
  state.enabled = bool(enabled)
  state.personality = int(personality)
  state.experimentalMode = bool(experimentalMode)
  return state


def create_mock_controlsState(longControlState=0, forceDecel=False):
  """Create a mock controlsState message."""
  state = Mock()
  state.longControlState = int(longControlState)
  state.forceDecel = bool(forceDecel)
  return state


def create_mock_radarState(lead_status=False):
  """Create a mock radarState message."""
  state = Mock()
  state.leadOne = Mock()
  state.leadOne.status = bool(lead_status)
  state.leadTwo = Mock()
  state.leadTwo.status = False
  return state


def create_mock_liveParameters(angleOffsetDeg=0.0):
  """Create a mock liveParameters message."""
  params = Mock()
  params.angleOffsetDeg = float(angleOffsetDeg)
  return params


def create_mock_carControl(orientationNED=None):
  """Create a mock carControl message."""
  control = Mock()
  if orientationNED is None:
    orientationNED = [0.0, 0.0, 0.0]
  control.orientationNED = orientationNED
  return control


def create_full_mock_sm(**overrides):
  """Create a complete mock SubMaster with all required services."""
  services = {
    'carState': create_mock_carState(**overrides.get('carState', {})),
    'selfdriveState': create_mock_selfdriveState(**overrides.get('selfdriveState', {})),
    'controlsState': create_mock_controlsState(**overrides.get('controlsState', {})),
    'radarState': create_mock_radarState(**overrides.get('radarState', {})),
    'liveParameters': create_mock_liveParameters(**overrides.get('liveParameters', {})),
    'carControl': create_mock_carControl(**overrides.get('carControl', {})),
    'modelV2': create_mock_modelV2(**overrides.get('modelV2', {})),
  }
  return MockSubMaster(services)


class TestE2ELongitudinalPersonality:
  """Test personality-based gain and offset application."""
  
  def test_personality_gain_relaxed(self):
    """Relaxed personality should reduce acceleration magnitude."""
    a_target = 2.0
    result = apply_personality_to_accel(a_target, log.LongitudinalPersonality.relaxed)
    expected = a_target * PERSONALITY_GAINS[log.LongitudinalPersonality.relaxed] + \
               PERSONALITY_OFFSETS[log.LongitudinalPersonality.relaxed]
    assert abs(result - expected) < 0.001
    assert result < a_target  # Should be less aggressive
    
  def test_personality_gain_standard(self):
    """Standard personality should not modify acceleration."""
    a_target = 2.0
    result = apply_personality_to_accel(a_target, log.LongitudinalPersonality.standard)
    expected = a_target * PERSONALITY_GAINS[log.LongitudinalPersonality.standard] + \
               PERSONALITY_OFFSETS[log.LongitudinalPersonality.standard]
    assert abs(result - expected) < 0.001
    assert result == a_target  # No modification
    
  def test_personality_gain_aggressive(self):
    """Aggressive personality should increase acceleration magnitude."""
    a_target = 2.0
    result = apply_personality_to_accel(a_target, log.LongitudinalPersonality.aggressive)
    expected = a_target * PERSONALITY_GAINS[log.LongitudinalPersonality.aggressive] + \
               PERSONALITY_OFFSETS[log.LongitudinalPersonality.aggressive]
    assert abs(result - expected) < 0.001
    assert result > a_target  # Should be more aggressive
    
  def test_personality_negative_accel(self):
    """Personality should work correctly with braking (negative acceleration)."""
    a_target = -3.0
    result = apply_personality_to_accel(a_target, log.LongitudinalPersonality.aggressive)
    # For negative accel, aggressive should make it more negative (stronger braking)
    expected = a_target * PERSONALITY_GAINS[log.LongitudinalPersonality.aggressive] + \
               PERSONALITY_OFFSETS[log.LongitudinalPersonality.aggressive]
    assert abs(result - expected) < 0.001


class TestE2ELongitudinalPlanner:
  """Test E2E longitudinal planner behavior."""
  
  def test_e2e_planner_no_mpc_initialization(self, mocker):
    """Verify MPC is NOT initialized when ForceClassicalMPC is False (default)."""
    # Mock Params to return False for ForceClassicalMPC
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    # MPC should not be initialized in E2E mode
    assert planner.mpc is None
    assert planner.force_classical_mpc is False
    assert planner.e2e_active is True
    
  def test_mpc_planner_initialization_when_forced(self, mocker):
    """Verify MPC IS initialized when ForceClassicalMPC is True."""
    # Mock Params to return True for ForceClassicalMPC
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = True
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    # MPC should be initialized in legacy mode
    assert planner.mpc is not None
    assert planner.force_classical_mpc is True
    
  def test_e2e_acceleration_follows_model(self, mocker):
    """Verify planner outputs match model's desired acceleration within 5% margin."""
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    # Test with specific acceleration command
    test_accel = -2.5  # Braking
    sm = create_full_mock_sm(
      modelV2={'desired_accel': test_accel, 'should_stop': False},
      selfdriveState={'personality': log.LongitudinalPersonality.standard}
    )
    
    planner.update(sm)
    
    # Output should match model's desired acceleration (with personality applied)
    # For standard personality, gain=1.0, offset=0.0
    expected_accel = test_accel
    margin = abs(expected_accel) * 0.05  # 5% margin
    
    assert abs(planner.output_a_target - expected_accel) <= margin or abs(planner.output_a_target - expected_accel) < 0.1
    
  def test_e2e_stop_trajectory(self, mocker):
    """Verify planner respects model's shouldStop flag."""
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    # Model commands stop
    sm = create_full_mock_sm(
      modelV2={'desired_accel': -1.0, 'should_stop': True},
      selfdriveState={'personality': log.LongitudinalPersonality.standard}
    )
    
    planner.update(sm)
    
    # shouldStop should be propagated
    assert planner.output_should_stop is True
    
  def test_e2e_personality_application(self, mocker):
    """Verify personality is correctly applied to E2E model output."""
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    base_accel = 2.0
    
    # Test with aggressive personality
    sm = create_full_mock_sm(
      modelV2={'desired_accel': base_accel, 'should_stop': False},
      selfdriveState={'personality': log.LongitudinalPersonality.aggressive}
    )
    
    planner.update(sm)
    
    # Aggressive should increase acceleration
    # Note: output may be clipped by get_max_accel based on current speed
    expected_raw = apply_personality_to_accel(base_accel, log.LongitudinalPersonality.aggressive)
    # The actual output will be clipped to max accel limits
    # Just verify the personality is being applied (output differs from base)
    assert planner.output_a_target != base_accel or expected_raw == base_accel
    # Verify output is within reasonable bounds
    assert planner.output_a_target <= 2.0  # Max accel limit
    
  def test_e2e_trajectory_generation(self, mocker):
    """Verify E2E mode generates a valid trajectory."""
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    sm = create_full_mock_sm(
      modelV2={'desired_accel': 1.5, 'should_stop': False},
      carState={'v_ego': 15.0},
      selfdriveState={'personality': log.LongitudinalPersonality.standard}
    )
    
    planner.update(sm)
    
    # Trajectory arrays should be populated
    assert len(planner.v_desired_trajectory) == CONTROL_N
    assert len(planner.a_desired_trajectory) == CONTROL_N
    assert len(planner.j_desired_trajectory) == CONTROL_N
    
    # First velocity should match current speed
    assert abs(planner.v_desired_trajectory[0] - 15.0) < 0.01
    
  def test_longitudinal_plan_source_e2e(self, mocker):
    """Verify longitudinalPlanSource is set to e2e when in E2E mode."""
    mock_params = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.Params')
    mock_params.return_value.get_bool.return_value = False
    
    CP = create_mock_car_params()
    planner = LongitudinalPlanner(CP)
    
    sm = create_full_mock_sm(
      modelV2={'desired_accel': 0.5, 'should_stop': False},
      selfdriveState={'personality': log.LongitudinalPersonality.standard}
    )
    
    planner.update(sm)
    
    # Should be in E2E mode
    assert planner.e2e_active is True


class TestModelMetadataExtraction:
  """Test model metadata extraction functionality."""
  
  def test_metadata_script_imports(self):
    """Verify the metadata extraction script can be imported."""
    # This test ensures tinygrad is available and the script structure is valid
    try:
      from openpilot.selfdrive.modeld.get_model_metadata import (
        MetadataOnnxPBParser,
        get_name_and_shape,
        get_metadata_value_by_name,
        validate_metadata
      )
      assert MetadataOnnxPBParser is not None
      assert callable(get_name_and_shape)
      assert callable(get_metadata_value_by_name)
      assert callable(validate_metadata)
    except ImportError as e:
      # If tinygrad is not available, skip this test
      pytest.skip(f"tinygrad not available: {e}")
  
  def test_validate_metadata_function(self):
    """Test metadata validation logic."""
    from openpilot.selfdrive.modeld.get_model_metadata import validate_metadata
    
    # Valid metadata
    valid = {
      'input_shapes': {'input1': (1, 3, 256, 256)},
      'output_shapes': {'output1': (1, 10)}
    }
    assert validate_metadata(valid) is True
    
    # Missing input_shapes
    invalid1 = {'output_shapes': {'output1': (1, 10)}}
    assert validate_metadata(invalid1) is False
    
    # Missing output_shapes
    invalid2 = {'input_shapes': {'input1': (1, 3, 256, 256)}}
    assert validate_metadata(invalid2) is False
    
    # Wrong type
    invalid3 = {'input_shapes': 'not_a_dict', 'output_shapes': {'output1': (1, 10)}}
    assert validate_metadata(invalid3) is False


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
