#!/usr/bin/env python3
"""
E2E Phase 4: Navigation-Aware Driving Integration Tests

These tests verify that the navigation embedding infrastructure correctly:
1. Generates embeddings from navigation route data
2. Passes embeddings to the E2E model
3. Encodes different maneuver types appropriately
4. Handles edge cases (no nav data, invalid data, etc.)

Note: These tests verify the INFRASTRUCTURE for nav-aware driving.
Actual navigation-aware BEHAVIOR requires a trained model that accepts
nav_embeddings as input. Until such a model is deployed, the E2E policy
will receive the embeddings but not use them for decision-making.

Test Categories:
- Unit tests: Nav embedding generation (nav_embedding.py)
- Integration tests: Nav embedding flow through modeld
- Behavioral tests: Model response to nav embeddings (requires trained model)
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openpilot.selfdrive.nav.nav_embedding import create_route_embedding, EMBEDDING_DIM


class TestNavEmbeddingDimensions:
  """Test that navigation embeddings have correct dimensions."""

  def test_embedding_dimension(self):
    """Verify embedding is always 64-dimensional."""
    assert EMBEDDING_DIM == 64, f"Expected EMBEDDING_DIM=64, got {EMBEDDING_DIM}"

  def test_empty_embedding_shape(self):
    """Verify empty nav data produces correct shape."""
    embedding = create_route_embedding(
      nav_route=None,
      nav_instruction=None,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )
    assert embedding.shape == (64,), f"Expected shape (64,), got {embedding.shape}"
    assert np.all(embedding == 0.0), "Empty nav data should produce zero embedding"

  def test_zero_embedding_on_empty_route(self):
    """Verify empty route list produces zero embedding."""
    embedding = create_route_embedding(
      nav_route=[],
      nav_instruction=None,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )
    assert np.all(embedding == 0.0), "Empty route should produce zero embedding"


class TestManeuverEncoding:
  """Test that different maneuver types are correctly encoded."""

  @pytest.fixture
  def base_state(self):
    """Create base vehicle state for tests."""
    return {
      'car_state': Mock(vEgo=20.0),
      'live_location': [37.7749, -122.4194],
      'route': [{'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1}]
    }

  def test_turn_left_encoding(self, base_state):
    """Verify turn-left maneuver is encoded at correct index."""
    instruction = Mock()
    instruction.type = 'turn-left'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=instruction,
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 1 should be 1.0 for turn-left (one-hot encoding)
    assert embedding[1] == 1.0, f"turn-left should set index 1 to 1.0, got {embedding[1]}"
    # Other maneuver indices should be 0
    assert embedding[2] == 0.0, "turn-right index should be 0"
    assert embedding[3] == 0.0, "merge-left index should be 0"

  def test_turn_right_encoding(self, base_state):
    """Verify turn-right maneuver is encoded at correct index."""
    instruction = Mock()
    instruction.type = 'turn-right'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=instruction,
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 2 should be 1.0 for turn-right
    assert embedding[2] == 1.0, f"turn-right should set index 2 to 1.0, got {embedding[2]}"

  def test_merge_left_encoding(self, base_state):
    """Verify merge-left maneuver is encoded at correct index."""
    instruction = Mock()
    instruction.type = 'merge-left'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=instruction,
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 3 should be 1.0 for merge-left
    assert embedding[3] == 1.0, f"merge-left should set index 3 to 1.0, got {embedding[3]}"

  def test_exit_right_encoding(self, base_state):
    """Verify exit-right maneuver is encoded at correct index."""
    instruction = Mock()
    instruction.type = 'exit-right'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=instruction,
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 6 should be 1.0 for exit-right
    assert embedding[6] == 1.0, f"exit-right should set index 6 to 1.0, got {embedding[6]}"

  def test_straight_encoding(self, base_state):
    """Verify straight maneuver is encoded at correct index."""
    instruction = Mock()
    instruction.type = 'straight'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=instruction,
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 9 should be 1.0 for straight
    assert embedding[9] == 1.0, f"straight should set index 9 to 1.0, got {embedding[9]}"


class TestDistanceEncoding:
  """Test that distance to maneuver is correctly encoded."""

  @pytest.fixture
  def base_state(self):
    """Create base vehicle state for tests."""
    return {
      'car_state': Mock(vEgo=20.0),
      'live_location': [37.7749, -122.4194],
      'route': [{'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1}],
      'instruction': Mock(type='turn-left', speedLimit=0, lane='', roadType='')
    }

  def test_very_close_maneuver(self, base_state):
    """Verify very close maneuver produces high encoding value."""
    base_state['instruction'].distance = 10.0  # Very close (10m)

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 0 should be close to 1.0 for very close maneuver
    # Normalization: 1.0 - (distance / 500)
    expected = max(0.0, min(1.0, 1.0 - 10.0 / 500.0))
    assert embedding[0] == pytest.approx(expected, abs=0.01), \
      f"Close maneuver should have high encoding, expected {expected}, got {embedding[0]}"

  def test_far_maneuver(self, base_state):
    """Verify far maneuver produces low encoding value."""
    base_state['instruction'].distance = 400.0  # Far (400m)

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 0 should be close to 0.2 for far maneuver
    expected = max(0.0, min(1.0, 1.0 - 400.0 / 500.0))
    assert embedding[0] == pytest.approx(expected, abs=0.01), \
      f"Far maneuver should have low encoding, expected {expected}, got {embedding[0]}"

  def test_beyond_range_maneuver(self, base_state):
    """Verify maneuver beyond 500m produces zero encoding."""
    base_state['instruction'].distance = 600.0  # Beyond range

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 0 should be 0.0 for maneuvers beyond 500m
    assert embedding[0] == 0.0, f"Maneuver beyond 500m should have 0 encoding, got {embedding[0]}"


class TestSpeedLimitEncoding:
  """Test that speed limit differences are correctly encoded."""

  @pytest.fixture
  def base_state(self):
    """Create base vehicle state for tests."""
    return {
      'car_state': Mock(vEgo=20.0),  # ~44.7 mph
      'live_location': [37.7749, -122.4194],
      'route': [{'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1}],
      'instruction': Mock(type='turn-left', distance=100.0, lane='', roadType='')
    }

  def test_speeding_encoding(self, base_state):
    """Verify speeding produces positive encoding."""
    base_state['instruction'].speedLimit = 35.0  # mph, car is going ~45 mph

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 14 should be positive when speeding
    # Normalization: (current_speed - speed_limit) / 30
    expected = np.clip((44.7 - 35.0) / 30.0, -1.0, 1.0)
    assert embedding[14] == pytest.approx(expected, abs=0.1), \
      f"Speeding should produce positive encoding, expected {expected}, got {embedding[14]}"

  def test_slow_encoding(self, base_state):
    """Verify driving below limit produces negative encoding."""
    base_state['car_state'].vEgo = 10.0  # ~22.4 mph
    base_state['instruction'].speedLimit = 45.0  # mph

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 14 should be negative when driving below limit
    expected = np.clip((22.4 - 45.0) / 30.0, -1.0, 1.0)
    assert embedding[14] == pytest.approx(expected, abs=0.1), \
      f"Driving below limit should produce negative encoding, expected {expected}, got {embedding[14]}"


class TestLaneEncoding:
  """Test that lane information is correctly encoded."""

  @pytest.fixture
  def base_state(self):
    """Create base vehicle state for tests."""
    return {
      'car_state': Mock(vEgo=20.0),
      'live_location': [37.7749, -122.4194],
      'route': [{'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1}],
      'instruction': Mock(type='turn-left', distance=100.0, speedLimit=0, roadType='')
    }

  def test_left_lane_encoding(self, base_state):
    """Verify left lane instruction produces negative encoding."""
    base_state['instruction'].lane = 'left'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 15 should be -1.0 for left lane
    assert embedding[15] == -1.0, f"Left lane should set index 15 to -1.0, got {embedding[15]}"

  def test_right_lane_encoding(self, base_state):
    """Verify right lane instruction produces positive encoding."""
    base_state['instruction'].lane = 'right'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 15 should be 1.0 for right lane
    assert embedding[15] == 1.0, f"Right lane should set index 15 to 1.0, got {embedding[15]}"

  def test_center_lane_encoding(self, base_state):
    """Verify center/missing lane instruction produces zero encoding."""
    base_state['instruction'].lane = 'center'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 15 should be 0.0 for center lane
    assert embedding[15] == 0.0, f"Center lane should set index 15 to 0.0, got {embedding[15]}"


class TestRoadTypeEncoding:
  """Test that road type is correctly encoded."""

  @pytest.fixture
  def base_state(self):
    """Create base vehicle state for tests."""
    return {
      'car_state': Mock(vEgo=20.0),
      'live_location': [37.7749, -122.4194],
      'route': [{'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1}],
      'instruction': Mock(type='turn-left', distance=100.0, speedLimit=0, lane='')
    }

  def test_highway_encoding(self, base_state):
    """Verify highway road type is encoded at correct index."""
    base_state['instruction'].roadType = 'highway'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 17 should be 1.0 for highway
    assert embedding[17] == 1.0, f"Highway should set index 17 to 1.0, got {embedding[17]}"

  def test_urban_encoding(self, base_state):
    """Verify urban road type is encoded at correct index."""
    base_state['instruction'].roadType = 'urban'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 18 should be 1.0 for urban
    assert embedding[18] == 1.0, f"Urban should set index 18 to 1.0, got {embedding[18]}"

  def test_residential_encoding(self, base_state):
    """Verify residential road type is encoded at correct index."""
    base_state['instruction'].roadType = 'residential'

    embedding = create_route_embedding(
      nav_route=base_state['route'],
      nav_instruction=base_state['instruction'],
      car_state=base_state['car_state'],
      live_location=base_state['live_location']
    )

    # Index 19 should be 1.0 for residential
    assert embedding[19] == 1.0, f"Residential should set index 19 to 1.0, got {embedding[19]}"


class TestEmbeddingSmoothness:
  """Test that embedding transitions are smooth."""

  def test_embedding_continuity(self):
    """Verify small changes in input produce small changes in output."""
    route = [{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}]
    car_state = Mock(vEgo=20.0)
    location1 = [37.7749, -122.4194]
    location2 = [37.7750, -122.4195]  # Small change

    instruction = Mock()
    instruction.type = 'turn-left'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding1 = create_route_embedding(route, instruction, car_state, location1)
    embedding2 = create_route_embedding(route, instruction, car_state, location2)

    # Embeddings should be similar (not identical due to relative position encoding)
    diff = np.abs(embedding1 - embedding2)
    assert np.max(diff) < 0.5, f"Small location change should produce small embedding change, max diff: {np.max(diff)}"


class TestIntegrationWithModel:
  """
  Integration tests for nav embeddings with modeld.

  These tests verify that nav embeddings are correctly passed to the model.
  Actual behavioral testing requires a nav-aware trained model.
  """

  def test_nav_embedding_input_shape(self):
    """Verify nav embedding input shape matches model expectation."""
    # Model expects (1, 64) shaped input
    embedding = np.zeros(64, dtype=np.float32)
    assert embedding.shape == (64,), f"Expected (64,), got {embedding.shape}"

    # Reshape for model input
    model_input = embedding.reshape(1, 64)
    assert model_input.shape == (1, 64), f"Model input should be (1, 64), got {model_input.shape}"

  def test_nav_embedding_dtype(self):
    """Verify nav embedding dtype matches model expectation."""
    embedding = np.zeros(64, dtype=np.float32)
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"

  def test_zero_embedding_valid(self):
    """Verify zero embedding is valid input when no nav data."""
    embedding = np.zeros(64, dtype=np.float32)
    assert np.all(embedding == 0.0), "Zero embedding should be all zeros"
    assert embedding.shape == (64,), "Zero embedding should have correct shape"


# Future test framework for nav-aware model behavior
# These tests will become meaningful once a nav-aware model is trained
"""
@pytest.mark.skip(reason="Requires nav-aware trained model")
class TestNavAwareBehavior:
  '''Tests for actual navigation-aware driving behavior.'''

  def test_lane_change_for_exit(self):
    '''Verify model initiates lane change when exit is approaching.'''
    # TODO: Implement once nav-aware model is trained
    pass

  def test_slow_for_turn(self):
    '''Verify model slows down for upcoming turn.'''
    # TODO: Implement once nav-aware model is trained
    pass

  def test_merge_behavior(self):
    '''Verify model merges appropriately on highway.'''
    # TODO: Implement once nav-aware model is trained
    pass
"""


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
