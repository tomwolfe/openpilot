import importlib
import numpy as np
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_meas_steer_torque_limits
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

class LatControlE2E:
  def __init__(self, CP, CI):
    self.CP = CP
    self.CI = CI
    # Smoothing for raw torque predictions
    self.torque_filter = FirstOrderFilter(0.0, 0.1, DT_CTRL)
    self.apply_torque_last = 0.0

    # Step 5: Dynamically import car-specific CarControllerParams
    try:
      brand = CP.brand
      values_module = importlib.import_module(f"opendbc.car.{brand}.values")
      self.params = values_module.CarControllerParams(CP)
    except (ImportError, AttributeError, TypeError):
      self.params = None

  def reset(self):
    self.torque_filter.x = 0.0
    self.apply_torque_last = 0.0

  def update(self, active, CS, VM, lp, steer_limited, model_torque, model_angle, curvature_limited, lat_delay):
    if not active:
      self.reset()
      return 0.0, 0.0, {}

    # Bypass PID/Torque controllers and map model_torque directly
    # Torque is usually -1.0 to 1.0 normalized
    steer_raw = self.torque_filter.update(model_torque)
    
    # Step 5: Enforce Safety Limits natively
    if self.params is not None:
      # Convert normalized steer to integer torque
      # Most cars use STEER_MAX to scale the normalized torque
      steer_max = getattr(self.params, "STEER_MAX", 1.0)
      apply_torque = int(round(steer_raw * steer_max))
      
      # Use the appropriate limit function based on what's available in params
      # This ensures we follow the specific car's safety logic (e.g., GM vs Toyota)
      driver_torque = CS.steeringTorque
      if hasattr(self.params, "STEER_DRIVER_ALLOWANCE"):
        apply_torque = apply_driver_steer_torque_limits(apply_torque, int(self.apply_torque_last), driver_torque, self.params)
      elif hasattr(self.params, "STEER_ERROR_MAX"):
        # For Toyota and others using meas torque limits
        # Note: CS.steeringTorque is used as a proxy for motor torque if needed
        apply_torque = apply_meas_steer_torque_limits(apply_torque, int(self.apply_torque_last), driver_torque, self.params)
      else:
        # Fallback to simple rate limit if neither is available
        delta_up = getattr(self.params, "STEER_DELTA_UP", steer_max)
        delta_down = getattr(self.params, "STEER_DELTA_DOWN", steer_max)
        apply_torque = np.clip(apply_torque, self.apply_torque_last - delta_down, self.apply_torque_last + delta_up)
        apply_torque = np.clip(apply_torque, -steer_max, steer_max)
      
      self.apply_torque_last = apply_torque
      steer = apply_torque / float(steer_max)
    else:
      steer = steer_raw
    
    # model_angle is the predicted absolute steering angle
    steering_angle_deg = model_angle

    return float(steer), float(steering_angle_deg), {"model_torque": steer, "model_angle": steering_angle_deg}
