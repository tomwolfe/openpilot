"""
Validation tests for Phase 1 E2E 1.0 Policy Schema.

Tests the new multi-hypothesis policy output format:
- PolicyHypothesis struct in cereal/log.capnp
- Policy class in selfdrive/modeld/constants.py
"""

import numpy as np
from cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan, Policy


class TestPolicyConstants:
  """Test Policy-related constants."""

  def test_plan_hypotheses_count(self):
    """Verify PLAN_HYPOTHESES_COUNT is set to 5."""
    assert ModelConstants.PLAN_HYPOTHESES_COUNT == 5
    assert ModelConstants.PLAN_HYPOTHESES_COUNT == ModelConstants.PLAN_MHP_N

  def test_policy_n_hypotheses(self):
    """Verify Policy.N_HYPOTHESES matches constant."""
    assert Policy.N_HYPOTHESES == ModelConstants.PLAN_HYPOTHESES_COUNT


class TestPolicySlices:
  """Test Policy slice utilities."""

  def test_policy_slices_match_plan(self):
    """Verify Policy slices are consistent with Plan slices."""
    assert Policy.POSITION == Plan.POSITION
    assert Policy.VELOCITY == Plan.VELOCITY
    assert Policy.ACCELERATION == Plan.ACCELERATION
    assert Policy.T_FROM_CURRENT_EULER == Plan.T_FROM_CURRENT_EULER
    assert Policy.ORIENTATION_RATE == Plan.ORIENTATION_RATE

  def test_get_hypothesis_slice(self):
    """Test hypothesis slice calculation for flattened tensor."""
    # Each hypothesis: IDX_N (33) timesteps * PLAN_WIDTH (15) values = 495 values
    values_per_hypothesis = ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH

    for i in range(ModelConstants.PLAN_HYPOTHESES_COUNT):
      slice_obj = Policy.get_hypothesis_slice(i)
      expected_start = i * values_per_hypothesis
      expected_end = (i + 1) * values_per_hypothesis
      assert slice_obj.start == expected_start
      assert slice_obj.stop == expected_end

  def test_get_timestep_slice(self):
    """Test timestep slice calculation within a hypothesis."""
    for hyp_idx in range(ModelConstants.PLAN_HYPOTHESES_COUNT):
      for t_idx in range(ModelConstants.IDX_N):
        slice_obj = Policy.get_timestep_slice(hyp_idx, t_idx)
        expected_start = (hyp_idx * ModelConstants.IDX_N + t_idx) * ModelConstants.PLAN_WIDTH
        expected_end = expected_start + ModelConstants.PLAN_WIDTH
        assert slice_obj.start == expected_start
        assert slice_obj.stop == expected_end
        assert slice_obj.stop - slice_obj.start == ModelConstants.PLAN_WIDTH

  def test_get_component_slice(self):
    """Test component slice calculation across all timesteps."""
    for hyp_idx in range(ModelConstants.PLAN_HYPOTHESES_COUNT):
      for _component_name, component_slice in [
        ('POSITION', Plan.POSITION),
        ('VELOCITY', Plan.VELOCITY),
        ('ACCELERATION', Plan.ACCELERATION),
      ]:
        indices = Policy.get_component_slice(hyp_idx, component_slice)
        # Should have IDX_N timesteps * component width values
        component_width = component_slice.stop - component_slice.start
        expected_len = ModelConstants.IDX_N * component_width
        assert len(indices) == expected_len


class TestPolicyHypothesisCapnp:
  """Test Cap'n Proto PolicyHypothesis struct."""

  def test_policy_hypothesis_struct_exists(self):
    """Verify PolicyHypothesis struct is defined in ModelDataV2."""
    # Create a new ModelDataV2 message
    msg = log.ModelDataV2.new_message()

    # Verify policy field exists and can be initialized
    msg.init('policy', 1)
    assert hasattr(msg, 'policy')
    assert len(msg.policy) == 1

  def test_policy_hypothesis_fields(self):
    """Verify PolicyHypothesis has required fields."""
    msg = log.ModelDataV2.new_message()

    # Initialize policy list with one hypothesis
    msg.init('policy', 1)

    hypothesis = msg.policy[0]

    # Verify all required fields exist
    assert hasattr(hypothesis, 'trajectory')
    assert hasattr(hypothesis, 'velocity')
    assert hasattr(hypothesis, 'acceleration')
    assert hasattr(hypothesis, 'probability')

  def test_policy_hypothesis_trajectory_structure(self):
    """Verify trajectory field has XYZTData structure."""
    msg = log.ModelDataV2.new_message()
    msg.init('policy', 1)

    hypothesis = msg.policy[0]

    # Set trajectory data
    n_points = ModelConstants.IDX_N
    hypothesis.trajectory.x = np.zeros(n_points).tolist()
    hypothesis.trajectory.y = np.zeros(n_points).tolist()
    hypothesis.trajectory.z = np.zeros(n_points).tolist()
    hypothesis.trajectory.t = np.zeros(n_points).tolist()

    assert len(hypothesis.trajectory.x) == n_points
    assert len(hypothesis.trajectory.y) == n_points
    assert len(hypothesis.trajectory.z) == n_points

  def test_probability_field_type(self):
    """Verify probability field is a float."""
    msg = log.ModelDataV2.new_message()
    msg.policy = [log.ModelDataV2.PolicyHypothesis.new_message()]

    hypothesis = msg.policy[0]
    hypothesis.probability = 0.5

    assert isinstance(hypothesis.probability, float)


class TestPolicyProbabilityValidation:
  """Test probability validation for multi-hypothesis output."""

  def test_probabilities_sum_to_one(self):
    """Verify that hypothesis probabilities sum to approximately 1.0."""
    msg = log.ModelDataV2.new_message()

    # Create 5 hypotheses with probabilities that sum to 1.0
    probabilities = [0.4, 0.3, 0.15, 0.1, 0.05]
    assert sum(probabilities) == 1.0

    # Initialize policy list with correct size
    msg.init('policy', len(probabilities))
    for i, prob in enumerate(probabilities):
      msg.policy[i].probability = prob

    # Verify sum
    total_prob = sum(h.probability for h in msg.policy)
    assert abs(total_prob - 1.0) < 1e-6

  def test_probabilities_normalized(self):
    """Test normalization of arbitrary probabilities."""
    msg = log.ModelDataV2.new_message()

    # Create hypotheses with unnormalized probabilities
    raw_probs = [4.0, 3.0, 1.5, 1.0, 0.5]
    total = sum(raw_probs)
    normalized_probs = [p / total for p in raw_probs]

    msg.init('policy', len(normalized_probs))
    for i, prob in enumerate(normalized_probs):
      msg.policy[i].probability = prob

    # Verify sum is ~1.0
    total_prob = sum(h.probability for h in msg.policy)
    assert abs(total_prob - 1.0) < 1e-6

  def test_single_dominant_hypothesis(self):
    """Test case where one hypothesis is dominant."""
    msg = log.ModelDataV2.new_message()

    # High confidence in first hypothesis
    probabilities = [0.95, 0.03, 0.01, 0.005, 0.005]

    msg.init('policy', len(probabilities))
    for i, prob in enumerate(probabilities):
      msg.policy[i].probability = prob

    total_prob = sum(h.probability for h in msg.policy)
    assert abs(total_prob - 1.0) < 1e-6
    assert msg.policy[0].probability > 0.9

  def test_uniform_probabilities(self):
    """Test case where all hypotheses are equally likely."""
    msg = log.ModelDataV2.new_message()

    # Uniform distribution
    n_hypotheses = ModelConstants.PLAN_HYPOTHESES_COUNT
    prob_each = 1.0 / n_hypotheses

    msg.init('policy', n_hypotheses)
    for i in range(n_hypotheses):
      msg.policy[i].probability = prob_each

    total_prob = sum(h.probability for h in msg.policy)
    assert abs(total_prob - 1.0) < 1e-5


class TestPolicyMessagePopulation:
  """Test full message population with multiple hypotheses."""

  def test_populate_full_policy_message(self):
    """Test populating a complete ModelDataV2 message with policy data."""
    msg = log.ModelDataV2.new_message()

    # Set basic fields
    msg.frameId = 100
    msg.timestampEof = 1234567890

    # Create 5 hypotheses with different trajectories
    n_points = ModelConstants.IDX_N
    probabilities = [0.4, 0.3, 0.15, 0.1, 0.05]

    msg.init('policy', len(probabilities))
    for i, prob in enumerate(probabilities):
      hypothesis = msg.policy[i]

      # Create slightly different trajectories for each hypothesis
      offset = i * 0.5  # meters offset between hypotheses

      hypothesis.trajectory.x = np.linspace(0, 100, n_points).tolist()
      hypothesis.trajectory.y = (np.linspace(0, 10, n_points) + offset).tolist()
      hypothesis.trajectory.z = np.zeros(n_points).tolist()
      hypothesis.trajectory.t = ModelConstants.T_IDXS

      hypothesis.velocity.x = (np.ones(n_points) * 15.0).tolist()  # 15 m/s
      hypothesis.velocity.y = np.zeros(n_points).tolist()
      hypothesis.velocity.z = np.zeros(n_points).tolist()
      hypothesis.velocity.t = ModelConstants.T_IDXS

      hypothesis.acceleration.x = np.zeros(n_points).tolist()
      hypothesis.acceleration.y = np.zeros(n_points).tolist()
      hypothesis.acceleration.z = np.zeros(n_points).tolist()
      hypothesis.acceleration.t = ModelConstants.T_IDXS

      hypothesis.probability = prob

    # Validate
    assert len(msg.policy) == ModelConstants.PLAN_HYPOTHESES_COUNT
    total_prob = sum(h.probability for h in msg.policy)
    assert abs(total_prob - 1.0) < 1e-6

    # Validate trajectory data
    for _i, h in enumerate(msg.policy):
      assert len(h.trajectory.x) == n_points
      assert len(h.velocity.x) == n_points
      assert len(h.acceleration.x) == n_points

  def test_backward_compatibility_with_plan(self):
    """Verify existing plan fields still work alongside new policy field."""
    msg = log.ModelDataV2.new_message()

    # Set old-style plan fields (for backward compatibility)
    n_points = ModelConstants.IDX_N
    msg.position.x = np.linspace(0, 100, n_points).tolist()
    msg.position.y = np.zeros(n_points).tolist()
    msg.position.z = np.zeros(n_points).tolist()
    msg.position.t = ModelConstants.T_IDXS

    msg.velocity.x = (np.ones(n_points) * 15.0).tolist()
    msg.velocity.y = np.zeros(n_points).tolist()
    msg.velocity.z = np.zeros(n_points).tolist()
    msg.velocity.t = ModelConstants.T_IDXS

    # Also set new policy field
    msg.init('policy', 1)
    msg.policy[0].probability = 1.0

    # Both should coexist
    assert len(msg.position.x) == n_points
    assert len(msg.policy) == 1
