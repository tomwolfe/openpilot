from cereal import car
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


class TestLongControl:
  """Tests for E2E direct longitudinal control."""

  def setup_method(self):
    self.CP = car.CarParams.new_message()
    self.lc = LongControl(self.CP)

  def test_off(self):
    """Test that controller outputs zero when inactive."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    accel = self.lc.update(active=False, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    assert accel == 0.0
    assert self.lc.long_control_state == LongCtrlState.off

  def test_gas_passthrough(self):
    """Test that gas predictions are passed through correctly."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    accel = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    # 0.5 gas * 2.0 max_accel = 1.0
    assert accel == 1.0
    assert self.lc.long_control_state == LongCtrlState.pid

  def test_brake_passthrough(self):
    """Test that brake predictions are passed through correctly."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 0.0, 'brake_pred': 0.5}
    accel = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    # 0.5 brake * -3.0 min_accel = -1.5
    assert accel == -1.5
    assert self.lc.long_control_state == LongCtrlState.pid

  def test_aeb_override(self):
    """Test that AEB override applies maximum braking."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0}
    accel = self.lc.update(active=True, CS=CS, model_output=model_output,
                           accel_limits=[-3.0, 2.0], aeb_override=-4.0)
    # AEB should override with maximum braking
    assert accel == -4.0
    assert self.lc.get_aeb_status() is True

  def test_aeb_not_triggered(self):
    """Test that AEB doesn't trigger when not needed."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 0.5, 'brake_pred': 0.0, 'crash_prob': 0.1, 'ttc_pred': 5.0}
    accel = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    assert accel == 1.0
    assert self.lc.get_aeb_status() is False

  def test_filter_smoothing(self):
    """Test that low-pass filter smooths commands."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    # First update with full gas
    model_output = {'gas_pred': 1.0, 'brake_pred': 0.0}
    accel1 = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    # Second update with zero gas - filter should smooth the transition
    model_output = {'gas_pred': 0.0, 'brake_pred': 0.0}
    accel2 = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    # accel2 should be less than full braking due to filter
    assert accel2 < 0.0
    assert accel2 > -3.0

  def test_clipping(self):
    """Test that output is clipped to vehicle limits."""
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    model_output = {'gas_pred': 1.5, 'brake_pred': 0.0}  # Invalid > 1.0
    accel = self.lc.update(active=True, CS=CS, model_output=model_output, accel_limits=[-3.0, 2.0])
    # Should be clipped to max_accel
    assert accel <= 2.0
