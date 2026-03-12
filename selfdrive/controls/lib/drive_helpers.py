import warnings
import numpy as np
from cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import index_function

CONTROL_N = 17

# E2E: Comfort limits are now learned by the neural network from human driving data.
# The Panda safety layer (opendbc/safety/) enforces absolute safety bounds.


# ============================================================================
# E2E Phase 1: MPC Constants (Deprecated)
# These constants are preserved for backward compatibility with tests and planners.
# The MPC controllers themselves are deprecated in favor of direct E2E actuation.
# ============================================================================

# Longitudinal MPC time indices (for trajectory interpolation)
LONG_MPC_N = 12
LONG_MPC_MAX_T = 10.0
T_IDXS_MPC = np.array([index_function(idx, max_val=LONG_MPC_MAX_T, max_idx=LONG_MPC_N) for idx in range(LONG_MPC_N + 1)])

# Lateral MPC constants
LAT_MPC_N = 32

# Longitudinal MPC constants (used in tests)
STOP_DISTANCE = 6.0
COMFORT_BRAKE = 2.5
CRUISE_MIN_ACCEL = -1.2
CRUISE_MAX_ACCEL = 1.6

# Lateral MPC constants
CAR_ROTATION_RADIUS = 0.0  # meters, rotation center offset (typically negligible for passenger cars)


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  """
  Get time gap for following distance calculation.
  
  Note: This is a legacy function. E2E model learns appropriate following
  distances from human data. Kept for test compatibility.
  """
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.8
  elif personality == log.LongitudinalPersonality.standard:
    return 1.4
  elif personality == log.LongitudinalPersonality.aggressive:
    return 1.0
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  """Low-pass filter for smoothing actuator commands."""
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val
