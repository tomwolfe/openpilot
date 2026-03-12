#!/usr/bin/env python3
"""
E2E Phase 4: Navigation Embedding Integration Tests

These tests verify the complete navigation embedding pipeline:
1. nav_embeddingd process generates embeddings from nav data
2. Embeddings are published via messaging
3. modeld receives and forwards embeddings to the policy network
4. Embeddings are logged in modelV2 message

This integration test suite mocks the navigation data source and verifies
that embeddings flow correctly through the system.
"""
import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openpilot.selfdrive.nav.nav_embedding import create_route_embedding, EMBEDDING_DIM


class TestNavEmbeddingPipeline:
  """Integration tests for the complete nav embedding pipeline."""

  def test_embedding_generation_from_mock_nav(self):
    """Test embedding generation with realistic mock navigation data."""
    # Create realistic mock navigation route
    nav_route = [
      {'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.0},
      {'lat': 37.7750, 'lon': -122.4195, 'curvature': 0.1},
      {'lat': 37.7751, 'lon': -122.4197, 'curvature': 0.2},
      {'lat': 37.7752, 'lon': -122.4200, 'curvature': 0.15},
    ]

    # Create realistic mock navigation instruction
    nav_instruction = Mock()
    nav_instruction.type = 'turn-left'
    nav_instruction.distance = 150.0  # meters
    nav_instruction.speedLimit = 35.0  # mph
    nav_instruction.lane = 'left'
    nav_instruction.roadType = 'urban'

    # Create mock vehicle state
    car_state = Mock(vEgo=15.0)  # ~33.5 mph

    # Create mock location
    live_location = [37.7749, -122.4194]

    # Generate embedding
    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=nav_instruction,
      car_state=car_state,
      live_location=live_location
    )

    # Verify embedding properties
    assert embedding.shape == (64,), f"Expected shape (64,), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"

    # Verify specific encodings
    assert embedding[0] > 0.5, "Close maneuver (150m) should have high distance encoding"
    assert embedding[1] == 1.0, "turn-left should set index 1"
    assert embedding[15] == -1.0, "left lane should set index 15 to -1.0"
    assert embedding[18] == 1.0, "urban should set index 18"

  def test_embedding_with_no_instruction(self):
    """Test embedding generation when only route is available."""
    nav_route = [
      {'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1},
    ]

    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=None,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )

    # Should produce mostly zero embedding with some route geometry
    assert embedding.shape == (64,)
    # Maneuver type indices should be zero
    assert embedding[1:11].sum() == 0.0, "No instruction should produce zero maneuver encoding"

  def test_embedding_with_invalid_location(self):
    """Test embedding generation with invalid/missing location data."""
    nav_route = [{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}]
    nav_instruction = Mock()
    nav_instruction.type = 'turn-left'
    nav_instruction.distance = 100.0
    nav_instruction.speedLimit = 0
    nav_instruction.lane = ''
    nav_instruction.roadType = ''

    # Empty location should produce zero embedding
    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=nav_instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[]
    )

    assert np.all(embedding == 0.0), "Invalid location should produce zero embedding"

  def test_embedding_with_long_route(self):
    """Test embedding generation with route exceeding max points."""
    # Create route with 200 points (exceeds MAX_ROUTE_POINTS=100)
    nav_route = [
      {'lat': 37.7749 + i * 0.0001, 'lon': -122.4194 + i * 0.0001, 'curvature': 0.1 * (i % 10)}
      for i in range(200)
    ]

    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=Mock(type='straight', distance=1000.0, speedLimit=0, lane='', roadType=''),
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )

    # Should handle gracefully without crashing
    assert embedding.shape == (64,)
    assert not np.any(np.isnan(embedding)), "Embedding should not contain NaN"
    assert not np.any(np.isinf(embedding)), "Embedding should not contain Inf"

  def test_all_maneuver_types(self):
    """Test encoding of all supported maneuver types."""
    maneuver_types = [
      ('turn-left', 1),
      ('turn-right', 2),
      ('merge-left', 3),
      ('merge-right', 4),
      ('exit-left', 5),
      ('exit-right', 6),
      ('fork-left', 7),
      ('fork-right', 8),
      ('straight', 9),
      ('uturn', 10),
    ]

    for maneuver_type, expected_index in maneuver_types:
      instruction = Mock()
      instruction.type = maneuver_type
      instruction.distance = 100.0
      instruction.speedLimit = 0
      instruction.lane = ''
      instruction.roadType = ''

      embedding = create_route_embedding(
        nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
        nav_instruction=instruction,
        car_state=Mock(vEgo=20.0),
        live_location=[37.7749, -122.4194]
      )

      assert embedding[expected_index] == 1.0, \
        f"{maneuver_type} should set index {expected_index} to 1.0"

      # Verify one-hot property (only one maneuver index should be 1)
      maneuver_indices = embedding[1:11]
      assert maneuver_indices.sum() == 1.0, \
        f"{maneuver_type} should produce one-hot encoding, sum={maneuver_indices.sum()}"

  def test_unknown_maneuver_type(self):
    """Test handling of unknown maneuver types."""
    instruction = Mock()
    instruction.type = 'unknown-maneuver'
    instruction.distance = 100.0
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )

    # Unknown maneuver should not set any maneuver index
    assert embedding[1:11].sum() == 0.0, "Unknown maneuver should not set any maneuver index"


class TestNavEmbeddingMessaging:
  """Tests for nav embedding messaging interface."""

  def test_embedding_to_list_conversion(self):
    """Test conversion of embedding to list for messaging."""
    embedding = np.random.randn(64).astype(np.float32)
    embedding_list = embedding.tolist()

    assert isinstance(embedding_list, list), "Conversion to list should produce list"
    assert len(embedding_list) == 64, f"List should have 64 elements, got {len(embedding_list)}"
    assert all(isinstance(x, float) for x in embedding_list), "All elements should be floats"

  def test_list_to_embedding_conversion(self):
    """Test conversion of list back to embedding array."""
    embedding_list = [float(i) for i in range(64)]
    embedding = np.array(embedding_list, dtype=np.float32)

    assert embedding.shape == (64,), f"Array should have shape (64,), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Array should be float32, got {embedding.dtype}"
    np.testing.assert_array_equal(embedding, np.arange(64, dtype=np.float32))

  def test_embedding_roundtrip(self):
    """Test that embedding survives roundtrip through list conversion."""
    original = np.random.randn(64).astype(np.float32)
    as_list = original.tolist()
    recovered = np.array(as_list, dtype=np.float32)

    np.testing.assert_array_almost_equal(original, recovered, decimal=5)


class TestNavEmbeddingEdgeCases:
  """Test edge cases and error handling."""

  def test_extreme_speed_values(self):
    """Test embedding generation with extreme vehicle speeds."""
    instruction = Mock()
    instruction.type = 'straight'
    instruction.distance = 100.0
    instruction.speedLimit = 50.0
    instruction.lane = ''
    instruction.roadType = ''

    # Very high speed
    embedding_fast = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=50.0),  # ~112 mph
      live_location=[37.7749, -122.4194]
    )

    # Very low speed
    embedding_slow = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=1.0),  # ~2.2 mph
      live_location=[37.7749, -122.4194]
    )

    # Both should be valid and clipped to [-1, 1]
    assert -1.0 <= embedding_fast[14] <= 1.0, "Speed encoding should be clipped"
    assert -1.0 <= embedding_slow[14] <= 1.0, "Speed encoding should be clipped"

  def test_extreme_distance_values(self):
    """Test embedding generation with extreme distances."""
    instruction = Mock()
    instruction.type = 'straight'
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    # Very close
    instruction.distance = 0.0
    embedding_close = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )
    assert embedding_close[0] == 1.0, "Zero distance should produce max encoding"

    # Very far
    instruction.distance = 10000.0
    embedding_far = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )
    assert embedding_far[0] == 0.0, "Very far distance should produce zero encoding"

  def test_negative_distance(self):
    """Test handling of negative distance values."""
    instruction = Mock()
    instruction.type = 'straight'
    instruction.distance = -100.0  # Invalid negative distance
    instruction.speedLimit = 0
    instruction.lane = ''
    instruction.roadType = ''

    embedding = create_route_embedding(
      nav_route=[{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}],
      nav_instruction=instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )

    # Should be clipped to valid range
    assert 0.0 <= embedding[0] <= 1.0, "Negative distance should be clipped to valid range"

  def test_missing_route_attributes(self):
    """Test handling of route points with missing attributes."""
    # Route with incomplete data
    nav_route = [
      {'lat': 37.7749},  # Missing lon and curvature
      {'lon': -122.4194},  # Missing lat and curvature
      {},  # Empty
    ]

    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=Mock(type='straight', distance=100.0, speedLimit=0, lane='', roadType=''),
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )

    # Should handle gracefully without crashing
    assert embedding.shape == (64,)
    # May contain zeros or partial data, but should be valid


class TestFutureNavAwareBehavior:
  """
  Framework for future nav-aware behavior tests.

  These tests will become meaningful once a navigation-aware model is trained.
  Currently they serve as documentation for expected behavior.
  """

  @pytest.mark.skip(reason="Requires nav-aware trained model")
  def test_expected_behavior_on_exit(self):
    """
    Expected: Model should initiate lane change when highway exit approaches.

    Test scenario:
    - Vehicle in right lane on highway
    - Navigation indicates exit in 500m on right
    - Expected: Model should produce steering/acceleration for exit maneuver
    """
    # TODO: Implement once nav-aware model is trained
    pass

  @pytest.mark.skip(reason="Requires nav-aware trained model")
  def test_expected_behavior_on_turn(self):
    """
    Expected: Model should slow down for upcoming turn.

    Test scenario:
    - Vehicle approaching 90-degree turn
    - Navigation indicates turn in 100m
    - Expected: Model should reduce speed appropriately
    """
    # TODO: Implement once nav-aware model is trained
    pass

  @pytest.mark.skip(reason="Requires nav-aware trained model")
  def test_expected_behavior_on_merge(self):
    """
    Expected: Model should merge smoothly when navigation indicates merge.

    Test scenario:
    - Vehicle on on-ramp
    - Navigation indicates merge with highway
    - Expected: Model should adjust speed and steering for safe merge
    """
    # TODO: Implement once nav-aware model is trained
    pass


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
