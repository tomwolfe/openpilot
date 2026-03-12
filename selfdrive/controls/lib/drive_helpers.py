import numpy as np
from openpilot.common.realtime import DT_MDL

CONTROL_N = 17

# E2E: Comfort limits are now learned by the neural network from human driving data.
# The Panda safety layer (opendbc/safety/) enforces absolute safety bounds.


def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  """Low-pass filter for smoothing actuator commands."""
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val
