import numpy as np

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)

class ModelConstants:
  # time and distance indices
  IDX_N = 33
  T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
  X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]
  LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]
  LEAD_T_OFFSETS = [0., 2., 4.]
  META_T_IDXS = [2., 4., 6., 8., 10.]

  # model inputs constants
  N_FRAMES = 2
  MODEL_RUN_FREQ = 20
  MODEL_CONTEXT_FREQ = 5 # "model_trained_fps"

  FEATURE_LEN = 512

  DESIRE_LEN = 8
  TRAFFIC_CONVENTION_LEN = 2
  LAT_PLANNER_STATE_LEN = 4
  LATERAL_CONTROL_PARAMS_LEN = 2
  PREV_DESIRED_CURV_LEN = 1

  # model outputs constants
  FCW_THRESHOLDS_5MS2 = np.array([.05, .05, .15, .15, .15], dtype=np.float32)
  FCW_THRESHOLDS_3MS2 = np.array([.7, .7], dtype=np.float32)
  FCW_5MS2_PROBS_WIDTH = 5
  FCW_3MS2_PROBS_WIDTH = 2

  DISENGAGE_WIDTH = 5
  POSE_WIDTH = 6
  WIDE_FROM_DEVICE_WIDTH = 3
  LEAD_WIDTH = 4
  LANE_LINES_WIDTH = 2
  ROAD_EDGES_WIDTH = 2
  PLAN_WIDTH = 15
  DESIRE_PRED_WIDTH = 8
  LAT_PLANNER_SOLUTION_WIDTH = 4
  DESIRED_CURV_WIDTH = 1

  NUM_LANE_LINES = 4
  NUM_ROAD_EDGES = 2

  LEAD_TRAJ_LEN = 6
  DESIRE_PRED_LEN = 4

  PLAN_MHP_N = 5
  LEAD_MHP_N = 2
  PLAN_MHP_SELECTION = 1
  LEAD_MHP_SELECTION = 3

  # Phase 1 E2E 1.0: Standardized policy schema
  PLAN_HYPOTHESES_COUNT = 5  # Number of trajectory hypotheses in policy output

  FCW_THRESHOLD_5MS2_HIGH = 0.15
  FCW_THRESHOLD_5MS2_LOW = 0.05
  FCW_THRESHOLD_3MS2 = 0.7

  CONFIDENCE_BUFFER_LEN = 5
  RYG_GREEN = 0.01165
  RYG_YELLOW = 0.06157

  POLY_PATH_DEGREE = 4

# model outputs slices
class Plan:
  POSITION = slice(0, 3)
  VELOCITY = slice(3, 6)
  ACCELERATION = slice(6, 9)
  T_FROM_CURRENT_EULER = slice(9, 12)
  ORIENTATION_RATE = slice(12, 15)

# E2E Phase 2: Direct actuator output slices
# These outputs represent the model's direct predictions for vehicle control
class E2EActuator:
  STEER_TORQUE = slice(0, 1)      # Steering torque [-1, 1]
  STEER_ANGLE = slice(1, 2)       # Steering angle in radians
  GAS = slice(2, 3)               # Gas pedal position [0, 1]
  BRAKE = slice(3, 4)             # Brake pedal position [0, 1]
  CRASH_PROB = slice(4, 5)        # Crash probability for AEB [0, 1]
  TTC = slice(5, 6)               # Time to collision in seconds

class Meta:
  ENGAGED = slice(0, 1)
  # next 2, 4, 6, 8, 10 seconds
  GAS_DISENGAGE = slice(1, 31, 6)
  BRAKE_DISENGAGE = slice(2, 31, 6)
  STEER_OVERRIDE = slice(3, 31, 6)
  HARD_BRAKE_3 = slice(4, 31, 6)
  HARD_BRAKE_4 = slice(5, 31, 6)
  HARD_BRAKE_5 = slice(6, 31, 6)
  # next 0, 2, 4, 6, 8, 10 seconds
  GAS_PRESS = slice(31, 55, 4)
  BRAKE_PRESS = slice(32, 55, 4)
  LEFT_BLINKER = slice(33, 55, 4)
  RIGHT_BLINKER = slice(34, 55, 4)


# Phase 1 E2E 1.0: Multi-hypothesis policy slices
# Provides indexing utilities for raw model output tensor containing multiple trajectory hypotheses
# Tensor shape: (N_HYPOTHESES, N_TIMESTEPS, PLAN_WIDTH) or flattened (N_HYPOTHESES * N_TIMESTEPS * PLAN_WIDTH)
class Policy:
  # Number of hypotheses in the policy output
  N_HYPOTHESES = ModelConstants.PLAN_HYPOTHESES_COUNT

  # Slice indices for extracting individual hypothesis components
  # Each hypothesis contains the same structure as Plan
  POSITION = Plan.POSITION
  VELOCITY = Plan.VELOCITY
  ACCELERATION = Plan.ACCELERATION
  T_FROM_CURRENT_EULER = Plan.T_FROM_CURRENT_EULER
  ORIENTATION_RATE = Plan.ORIENTATION_RATE

  @staticmethod
  def get_hypothesis_slice(hypothesis_idx: int) -> slice:
    """
    Returns a slice for extracting a specific hypothesis from a flattened tensor.
    Tensor layout: [hyp0_t0, hyp0_t1, ..., hyp0_tN, hyp1_t0, ..., hypN_tN]
    Each timestep has PLAN_WIDTH (15) values.
    """
    start = hypothesis_idx * ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH
    end = start + ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH
    return slice(start, end)

  @staticmethod
  def get_timestep_slice(hypothesis_idx: int, timestep_idx: int) -> slice:
    """
    Returns a slice for extracting a specific timestep from a specific hypothesis.
    Each timestep has PLAN_WIDTH (15) values.
    """
    start = (hypothesis_idx * ModelConstants.IDX_N + timestep_idx) * ModelConstants.PLAN_WIDTH
    end = start + ModelConstants.PLAN_WIDTH
    return slice(start, end)

  @staticmethod
  def get_component_slice(hypothesis_idx: int, component_slice: slice) -> np.ndarray:
    """
    Returns a slice for extracting a specific component (e.g., POSITION, VELOCITY)
    across all timesteps for a given hypothesis.
    """
    base = hypothesis_idx * ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH
    # Build list of indices for the component across all timesteps
    indices = []
    for t in range(ModelConstants.IDX_N):
      timestep_offset = t * ModelConstants.PLAN_WIDTH
      comp_start = base + timestep_offset + component_slice.start
      comp_end = base + timestep_offset + component_slice.stop
      indices.extend(range(comp_start, comp_end))
    # Return as a numpy array for advanced indexing
    return np.array(indices, dtype=np.int32)
