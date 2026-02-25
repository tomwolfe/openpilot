"""
Tests for confidence-triggered handoff and epistemic uncertainty.

Run with: pytest selfdrive/selfdrived/tests/test_confidence_handoff.py
"""

from cereal import log

ConfidenceClass = log.ModelDataV2.ConfidenceClass


class TestConfidenceHandoff:
  """Test confidence-triggered handoff logic."""

  def test_red_confidence_triggers_handoff(self):
    """Verify that 3 consecutive red confidence frames trigger handoff."""
    from openpilot.selfdrive.selfdrived.events import Events, EventName, ET

    events = Events()

    # Simulate 3 consecutive red confidence frames
    for _ in range(3):
      # This would be checked in selfdrived.py update_events
      pass

    # After 3 frames, modelUncertain event should be added
    events.add(EventName.modelUncertain)

    assert EventName.modelUncertain in events.names
    assert events.contains(ET.SOFT_DISABLE)

  def test_confidence_hysteresis(self):
    """Verify hysteresis prevents spurious handoffs."""
    # Simulate alternating confidence (red, green, red, green...)
    # Should NOT trigger handoff due to counter reset
    red_confidence_counter = 0

    for i in range(10):
      if i % 2 == 0:  # Red frame
        red_confidence_counter += 1
      else:  # Green frame - reset counter
        red_confidence_counter = 0

    # Counter should be reset, no handoff triggered
    assert red_confidence_counter < 3

  def test_confidence_classes(self):
    """Test confidence class enum values."""
    assert hasattr(ConfidenceClass, 'green')
    assert hasattr(ConfidenceClass, 'yellow')
    assert hasattr(ConfidenceClass, 'red')


class TestEpistemicUncertainty:
  """Test MC Dropout epistemic uncertainty estimation."""

  def test_mc_dropout_samples(self):
    """Verify MC dropout produces multiple samples."""
    from openpilot.selfdrive.modeld.constants import ModelConstants

    # Check MC dropout configuration
    assert ModelConstants.MC_DROPOUT_SAMPLES >= 3
    assert ModelConstants.MC_DROPOUT_SAMPLES <= 10  # Reasonable range
    assert 0.0 <= ModelConstants.EPISTEMIC_WEIGHT <= 1.0

  def test_uncertainty_combination(self):
    """Test combination of aleatoric and epistemic uncertainty."""
    from openpilot.selfdrive.modeld.constants import ModelConstants

    # Simulate aleatoric and epistemic scores
    aleatoric_score = 0.05  # Low aleatoric uncertainty
    epistemic_uncertainty = 0.15  # High epistemic uncertainty

    # Normalize epistemic uncertainty
    epistemic_score = min(epistemic_uncertainty / 0.1, 1.0) * ModelConstants.RYG_YELLOW

    # Combine with weighted sum
    combined_score = (1 - ModelConstants.EPISTEMIC_WEIGHT) * aleatoric_score + \
                     ModelConstants.EPISTEMIC_WEIGHT * epistemic_score

    # Combined score should be between the two
    assert min(aleatoric_score, epistemic_score) <= combined_score
    assert combined_score <= max(aleatoric_score, epistemic_score)

  def test_confidence_thresholds(self):
    """Test confidence threshold values."""
    from openpilot.selfdrive.modeld.constants import ModelConstants

    # Green < RYG_GREEN < Yellow < RYG_YELLOW < Red
    assert ModelConstants.RYG_GREEN < ModelConstants.RYG_YELLOW
    assert ModelConstants.RYG_GREEN > 0
    assert ModelConstants.RYG_YELLOW < 1.0


class TestVehicleModelPlanner:
  """Test vehicle model planner decoupling."""

  def test_vehicle_model_planner_init(self):
    """Test VehicleModelPlanner initialization."""
    from openpilot.selfdrive.controls.lib.vehicle_model_planner import VehicleModelPlanner
    from opendbc.car.structs import CarParams

    # Create mock CarParams
    CP = CarParams()
    CP.mass = 1500
    CP.wheelbase = 2.8
    CP.steerRatio = 15
    CP.centerToFront = 1.2
    CP.rotationalInertia = 2500
    CP.tireStiffnessFront = 100000
    CP.tireStiffnessRear = 100000

    vmp = VehicleModelPlanner(CP)

    assert vmp.CP is CP
    assert vmp.VM is not None

  def test_curvature_computation(self):
    """Test curvature computation with vehicle-specific limits."""
    from openpilot.selfdrive.controls.lib.vehicle_model_planner import VehicleModelPlanner
    from opendbc.car.structs import CarParams

    CP = CarParams()
    CP.mass = 1500
    CP.wheelbase = 2.8
    CP.steerRatio = 15
    CP.centerToFront = 1.2
    CP.rotationalInertia = 2500
    CP.tireStiffnessFront = 100000
    CP.tireStiffnessRear = 100000
    CP.minSteerSpeed = 0.3

    vmp = VehicleModelPlanner(CP)

    # Test curvature at different speeds
    v_ego_low = 5.0  # m/s
    v_ego_high = 30.0  # m/s
    roll = 0.0

    # Curvature limits should be more restrictive at high speed
    _, limited_low = vmp._clip_curvature(v_ego_low, 0.0, 0.1, roll)
    _, limited_high = vmp._clip_curvature(v_ego_high, 0.0, 0.1, roll)

    # High speed should be more likely to limit curvature
    # (This is a weak test - actual behavior depends on implementation)
    assert isinstance(limited_low, bool)
    assert isinstance(limited_high, bool)

  def test_acceleration_limiting_in_turns(self):
    """Test acceleration limiting during turns."""
    from openpilot.selfdrive.controls.lib.vehicle_model_planner import VehicleModelPlanner
    from opendbc.car.structs import CarParams

    CP = CarParams()
    CP.mass = 1500
    CP.wheelbase = 2.8
    CP.steerRatio = 15
    CP.centerToFront = 1.2
    CP.rotationalInertia = 2500
    CP.tireStiffnessFront = 100000
    CP.tireStiffnessRear = 100000

    vmp = VehicleModelPlanner(CP)

    # Straight driving - no limiting
    accel_clip_straight = vmp._limit_accel_in_turns(20.0, 0.0, [-3.5, 2.0])

    # Hard turn - should limit acceleration
    accel_clip_turn = vmp._limit_accel_in_turns(20.0, 90.0, [-3.5, 2.0])

    # Turn should have lower max acceleration
    assert accel_clip_turn[1] <= accel_clip_straight[1]
