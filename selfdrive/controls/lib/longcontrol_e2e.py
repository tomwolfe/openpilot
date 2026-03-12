import numpy as np
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

class LongControlE2E:
  def __init__(self, CP):
    self.CP = CP
    # Phase 2 E2E: Direct gas/brake mapping with low-pass filtering for smoothness
    self.gas_filter = FirstOrderFilter(0.0, 0.5, DT_CTRL)
    self.brake_filter = FirstOrderFilter(0.0, 0.2, DT_CTRL)

  def update(self, active, CS, sm):
    if not active:
      self.gas_filter.x = 0.0
      self.brake_filter.x = 0.0
      return 0.0, 0.0

    model_v2 = sm['modelV2']
    # Step 3: Vision-Based AEB logic
    # Extract crashProbability from the modelV2 capnp message
    CRASH_PROB_THRESHOLD = 0.85
    if model_v2.action.crashProbability > CRASH_PROB_THRESHOLD:
      return 0.0, 1.0  # (Gas = 0%, Brake = 100%)

    model_gas = model_v2.action.gas
    model_brake = model_v2.action.brake

    # Smooth the model's direct gas/brake predictions
    gas = self.gas_filter.update(model_gas)
    brake = self.brake_filter.update(model_brake)

    # Step 2: Return filtered gas and brake values as a tuple
    return float(gas), float(brake)
