from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

class LatControlE2E:
  def __init__(self, CP, CI):
    self.CP = CP
    self.CI = CI
    # Smoothing for raw torque predictions
    self.torque_filter = FirstOrderFilter(0.0, 0.1, DT_CTRL)

  def reset(self):
    self.torque_filter.x = 0.0

  def update(self, active, CS, VM, lp, steer_limited, model_torque, model_angle, curvature_limited, lat_delay):
    if not active:
      self.reset()
      return 0.0, 0.0, {}

    # Bypass PID/Torque controllers and map model_torque directly
    # Torque is usually -1.0 to 1.0 normalized
    steer = self.torque_filter.update(model_torque)
    
    # model_angle is the predicted absolute steering angle
    steering_angle_deg = model_angle

    return float(steer), float(steering_angle_deg), {"model_torque": steer, "model_angle": steering_angle_deg}
