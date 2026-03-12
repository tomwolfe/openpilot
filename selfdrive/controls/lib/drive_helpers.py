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


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  """
  Extract acceleration target from model's velocity/acceleration plan.
  
  E2E Phase 2: Removed vEgoStopping heuristic - the model learns appropriate
  stopping behavior implicitly from human driving data. The model should output
  a_target = 0 (or necessary holding brake pressure) when stopped.
  
  Classical controllers used this for state machine logic, but E2E controllers
  bypass this entirely by using direct actuator predictions (gas_pred, brake_pred).
  """
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  
  # E2E Phase 2: Return None for should_stop - model decides when to stop
  # Classical mode compatibility: return False to maintain legacy behavior
  return a_target, False

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
