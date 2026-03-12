from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol import LatControl


class TestLatControl:
  """Tests for E2E direct lateral control."""

  def setup_method(self):
    self.CP = car.CarParams.new_message()
    self.CP.steerLimitTimer = 1.0
    car_name = HONDA.HONDA_CIVIC
    CarInterface = interfaces[car_name]
    self.CI = CarInterface(self.CP)
    self.lac = LatControl(self.CP, self.CI, DT_CTRL)

  def test_off(self):
    """Test that controller outputs zero when inactive."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    model_output = {'steer_torque_pred': 0.5, 'steer_angle_pred': 0.1}
    torque, angle_deg, lac_log = self.lac.update(
      active=False, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    assert torque == 0.0
    assert angle_deg == 0.0
    assert lac_log.active is False

  def test_torque_passthrough(self):
    """Test that torque predictions are passed through correctly."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    model_output = {'steer_torque_pred': 0.5, 'steer_angle_pred': 0.0}
    torque, angle_deg, lac_log = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    # Torque should be close to 0.5 (may be filtered)
    assert 0.4 < torque < 0.6
    assert lac_log.active is True
    assert lac_log.outputTorque == float(torque)

  def test_angle_passthrough(self):
    """Test that angle predictions are passed through correctly."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    model_output = {'steer_torque_pred': 0.0, 'steer_angle_pred': 0.1}  # 0.1 radians
    torque, angle_deg, lac_log = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    # Angle should be close to 0.1 radians converted to degrees (~5.73 degrees)
    import math
    expected_deg = math.degrees(0.1)
    assert abs(angle_deg - expected_deg) < 1.0  # Allow for filter smoothing
    assert lac_log.active is True
    assert lac_log.steeringAngleDesiredDeg == float(angle_deg)

  def test_clipping(self):
    """Test that output is clipped to [-1, 1] range."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    model_output = {'steer_torque_pred': 2.0, 'steer_angle_pred': 0.0}  # Invalid > 1.0
    torque, angle_deg, lac_log = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    # Should be clipped to 1.0
    assert torque <= 1.0

  def test_filter_smoothing(self):
    """Test that low-pass filter smooths commands."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    # First update with full torque
    model_output = {'steer_torque_pred': 1.0, 'steer_angle_pred': 0.0}
    torque1, _, _ = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    # Second update with zero torque - filter should smooth the transition
    model_output = {'steer_torque_pred': 0.0, 'steer_angle_pred': 0.0}
    torque2, _, _ = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=model_output
    )
    # torque2 should be less than full due to filter
    assert torque2 < torque1
    assert torque2 > 0.0

  def test_no_model_output_warning(self):
    """Test that missing model output is handled gracefully."""
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=0.0, steeringPressed=False)
    torque, angle_deg, lac_log = self.lac.update(
      active=True, CS=CS, VM=None, params=None,
      steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False,
      lat_delay=0.1, model_output=None
    )
    # Should output zeros when no model output
    assert torque == 0.0
    assert angle_deg == 0.0
