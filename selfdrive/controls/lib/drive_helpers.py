import numpy as np
from openpilot.common.realtime import DT_MDL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
MAX_VEL_ERR = 5.0  # m/s

# E2E Phase 2: Comfort limits are now learned by the neural network from human driving data.
# The Panda safety layer (opendbc/safety/) enforces absolute safety bounds.
# Removed hardcoded heuristics:
#   - MAX_LATERAL_JERK = 5.0 (EU guideline, now learned)
#   - MAX_LATERAL_ACCEL_NO_ROLL = 3.0 (comfort limit, now learned)
#   - MAX_CURVATURE = 0.2 (artificial constraint, removed)
#   - clip_curvature() function (no longer needed, model learns appropriate limits)


def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  """Low-pass filter for smoothing actuator commands."""
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val


# E2E Phase 2: clip_curvature() removed - model learns appropriate curvature limits
# Classical controllers used this for comfort heuristic limiting.
# In true E2E, the neural network outputs direct actuator commands that implicitly
# respect comfort bounds learned from training data.
# Panda safety layer enforces absolute physical limits.


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL, vEgoStopping=0.05):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
    v_target_1sec = np.interp(action_t + 1.0, t_idxs, speeds)
  else:
    v_target = 0.0
    v_target_1sec = 0.0
    a_target = 0.0
  should_stop = (v_target < vEgoStopping and
                 v_target_1sec < vEgoStopping)
  return a_target, should_stop

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
