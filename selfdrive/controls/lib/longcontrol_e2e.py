import numpy as np
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

class LongControlE2E:
  def __init__(self, CP):
    self.CP = CP
    # Phase 2 E2E: Direct gas/brake mapping with low-pass filtering for smoothness
    self.gas_filter = FirstOrderFilter(0.0, 0.5, DT_CTRL)
    self.brake_filter = FirstOrderFilter(0.0, 0.2, DT_CTRL)

  def update(self, active, CS, model_gas, model_brake):
    if not active:
      self.gas_filter.x = 0.0
      self.brake_filter.x = 0.0
      return 0.0

    # Smooth the model's direct gas/brake predictions
    gas = self.gas_filter.update(model_gas)
    brake = self.brake_filter.update(model_brake)

    # Map to unified accel interface (-3.5 to 2.0 m/s^2 typical)
    # This is a simplified mapping for the prototype
    accel = gas - brake
    return float(accel)
