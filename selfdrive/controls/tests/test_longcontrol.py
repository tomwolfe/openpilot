from cereal import car
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


class TestLongControl:
  def setup_method(self):
    self.CP = car.CarParams.new_message()
    self.CP.longitudinalTuning.kpBP = [0.0, 10.0, 40.0]
    self.CP.longitudinalTuning.kpV = [1.2, 0.8, 0.5]
    self.CP.longitudinalTuning.kiBP = [0.0, 10.0, 40.0]
    self.CP.longitudinalTuning.kiV = [0.18, 0.12, 0.1]
    self.lc = LongControl(self.CP)

  def test_off(self):
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    accel = self.lc.update(active=False, CS=CS, a_target=1.0, should_stop=False, accel_limits=[-3.0, 2.0])
    assert accel == 0.0
    assert self.lc.long_control_state == LongCtrlState.off

  def test_pid_following(self):
    # Test accelerating
    CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)
    accel = self.lc.update(active=True, CS=CS, a_target=1.0, should_stop=False, accel_limits=[-3.0, 2.0])
    assert self.lc.long_control_state == LongCtrlState.pid
    assert accel > 0.0

    # Test decelerating
    accel = self.lc.update(active=True, CS=CS, a_target=-1.0, should_stop=False, accel_limits=[-3.0, 2.0])
    assert accel < 0.0

  def test_stopping_target(self):
    # E2E refactor: verify it handles stopping via a_target, not states
    CS = car.CarState.new_message(vEgo=0.5, aEgo=-0.1)
    # Target a strong deceleration to stop
    accel = self.lc.update(active=True, CS=CS, a_target=-1.5, should_stop=True, accel_limits=[-3.0, 2.0])
    assert self.lc.long_control_state == LongCtrlState.pid
    assert accel < -0.1  # Should be requesting more braking
