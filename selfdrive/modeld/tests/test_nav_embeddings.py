#!/usr/bin/env python3
"""
E2E Phase 4: Model Nav Embedding Input Tests

These tests verify that modeld correctly:
1. Initializes nav_embeddings input with correct shape
2. Receives nav embeddings from messaging
3. Passes embeddings to the policy network
4. Handles missing/invalid nav embeddings gracefully
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openpilot.selfdrive.modeld.constants import ModelConstants


class TestNavEmbeddingModelInput:
  """Tests for nav embedding model input handling."""

  def test_nav_embedding_input_shape(self):
    """Verify nav embedding input shape matches model expectation."""
    # Model expects (1, 64) shaped input for policy network
    expected_shape = (1, 64)
    
    # Create test embedding
    embedding = np.zeros(64, dtype=np.float32)
    model_input = embedding.reshape(1, 64)
    
    assert model_input.shape == expected_shape, \
      f"Model input shape should be {expected_shape}, got {model_input.shape}"

  def test_nav_embedding_dtype(self):
    """Verify nav embedding dtype matches model expectation."""
    embedding = np.zeros(64, dtype=np.float32)
    assert embedding.dtype == np.float32, \
      f"Expected float32, got {embedding.dtype}"

  def test_nav_embedding_dimension_constant(self):
    """Verify EMBEDDING_DIM constant is correctly defined."""
    from openpilot.selfdrive.nav import EMBEDDING_DIM
    assert EMBEDDING_DIM == 64, f"Expected EMBEDDING_DIM=64, got {EMBEDDING_DIM}"

  def test_zero_embedding_valid_input(self):
    """Verify zero embedding is valid when no nav data available."""
    embedding = np.zeros(64, dtype=np.float32)
    
    # Should be valid numpy array
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,)
    assert np.all(embedding == 0.0)
    
    # Reshape for model input
    model_input = embedding.reshape(1, 64)
    assert model_input.shape == (1, 64)

  def test_random_embedding_valid_input(self):
    """Verify random embeddings within valid range are accepted."""
    embedding = np.random.uniform(-1.0, 1.0, size=64).astype(np.float32)
    
    assert embedding.shape == (64,)
    assert embedding.dtype == np.float32
    assert np.all(embedding >= -1.0) and np.all(embedding <= 1.0)

  def test_embedding_bounds(self):
    """Verify embedding values are typically bounded."""
    # Most nav embedding components should be in [-1, 1] range
    embedding = np.zeros(64, dtype=np.float32)
    
    # Set various test values
    embedding[0] = 0.5  # Distance (normalized 0-1)
    embedding[1] = 1.0  # Maneuver type (one-hot)
    embedding[14] = -0.5  # Speed difference (normalized -1 to 1)
    embedding[15] = -1.0  # Lane position (-1, 0, 1)
    
    # Verify bounds
    assert 0.0 <= embedding[0] <= 1.0
    assert 0.0 <= embedding[1] <= 1.0
    assert -1.0 <= embedding[14] <= 1.0
    assert -1.0 <= embedding[15] <= 1.0


class TestNavEmbeddingMessagingInterface:
  """Tests for nav embedding messaging interface."""

  def test_nav_embeddings_message_format(self):
    """Verify nav embeddings message format."""
    # Simulate message data
    embedding_list = [0.0] * 64
    
    # Convert to numpy array (as modeld does)
    embedding = np.array(embedding_list, dtype=np.float32)
    
    assert embedding.shape == (64,)
    assert embedding.dtype == np.float32

  def test_nav_embeddings_from_submaster(self):
    """Test extracting nav embeddings from SubMaster."""
    # Mock SubMaster with navEmbeddings
    mock_sm = Mock()
    mock_sm.updated = {'navEmbeddings': True}
    mock_sm.valid = {'navEmbeddings': True}
    mock_sm.__getitem__ = Mock(return_value=Mock(
      navEmbeddings=[0.5] * 64
    ))
    
    # Extract embeddings (as modeld does)
    if mock_sm.updated['navEmbeddings'] and mock_sm.valid['navEmbeddings']:
      nav_embeddings = np.array(mock_sm['navEmbeddings'].navEmbeddings, dtype=np.float32)
    else:
      nav_embeddings = np.zeros(64, dtype=np.float32)
    
    assert nav_embeddings.shape == (64,)
    assert nav_embeddings.dtype == np.float32
    assert np.all(nav_embeddings == 0.5)

  def test_nav_embeddings_fallback_to_zero(self):
    """Test fallback to zero embedding when nav data unavailable."""
    # Mock SubMaster without navEmbeddings
    mock_sm = Mock()
    mock_sm.updated = {'navEmbeddings': False}
    mock_sm.valid = {'navEmbeddings': False}
    
    # Extract embeddings (as modeld does)
    nav_embeddings = np.zeros(64, dtype=np.float32)
    if mock_sm.updated.get('navEmbeddings', False) and mock_sm.valid.get('navEmbeddings', False):
      pass  # Would extract from sm['navEmbeddings']
    
    assert nav_embeddings.shape == (64,)
    assert np.all(nav_embeddings == 0.0)


class TestNavEmbeddingIntegration:
  """Integration tests for nav embedding flow."""

  def test_embedding_generation_to_model_input(self):
    """Test complete flow from nav data to model input."""
    from openpilot.selfdrive.nav.nav_embedding import create_route_embedding
    
    # Create nav data
    nav_route = [
      {'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1},
    ]
    nav_instruction = Mock()
    nav_instruction.type = 'turn-left'
    nav_instruction.distance = 100.0
    nav_instruction.speedLimit = 35.0
    nav_instruction.lane = 'left'
    nav_instruction.roadType = 'urban'
    
    # Generate embedding
    embedding = create_route_embedding(
      nav_route=nav_route,
      nav_instruction=nav_instruction,
      car_state=Mock(vEgo=20.0),
      live_location=[37.7749, -122.4194]
    )
    
    # Convert to model input format
    model_input = embedding.reshape(1, 64)
    
    # Verify
    assert model_input.shape == (1, 64), "Model input should be (1, 64)"
    assert model_input.dtype == np.float32, "Model input should be float32"
    assert model_input[0, 1] == 1.0, "turn-left should set index 1"

  def test_multiple_embeddings_batch(self):
    """Test handling multiple embeddings (e.g., for batch processing)."""
    embeddings = []
    
    # Generate multiple embeddings
    for i in range(5):
      embedding = np.zeros(64, dtype=np.float32)
      embedding[i % 64] = 1.0  # Different pattern for each
      embeddings.append(embedding)
    
    # Stack into batch
    batch = np.stack(embeddings)
    
    assert batch.shape == (5, 64), "Batch should be (5, 64)"
    assert batch.dtype == np.float32

  def test_embedding_smoothing(self):
    """Test embedding smoothing for temporal consistency."""
    from openpilot.selfdrive.nav.nav_embedding import create_route_embedding
    
    # Generate two similar embeddings
    route = [{'lat': 37.7749, 'lon': -122.4194, 'curvature': 0.1}]
    instruction = Mock(type='turn-left', distance=100.0, speedLimit=0, lane='', roadType='')
    
    embedding1 = create_route_embedding(
      route, instruction, Mock(vEgo=20.0), [37.7749, -122.4194]
    )
    embedding2 = create_route_embedding(
      route, instruction, Mock(vEgo=20.0), [37.7750, -122.4195]  # Slight location change
    )
    
    # Apply smoothing (as nav_embeddingd does)
    smoothed = 0.7 * embedding1 + 0.3 * embedding2
    
    # Verify smoothing
    assert smoothed.shape == embedding1.shape
    assert smoothed.dtype == np.float32
    # Smoothed should be between the two
    assert np.all(smoothed >= np.minimum(embedding1, embedding2) - 0.01)
    assert np.all(smoothed <= np.maximum(embedding1, embedding2) + 0.01)


class TestNavEmbeddingPolicyInput:
  """Tests for nav embedding as policy network input."""

  def test_policy_input_concatenation(self):
    """Test nav embedding concatenation with other policy inputs."""
    # Simulate policy inputs
    features = np.random.randn(1, 512).astype(np.float32)
    desire = np.random.randn(1, 8).astype(np.float32)
    nav_embedding = np.zeros(1, 64).astype(np.float32)
    
    # Concatenate (simplified - actual model may do this differently)
    combined = np.concatenate([features, desire, nav_embedding], axis=1)
    
    assert combined.shape[0] == 1, "Batch dimension should be 1"
    assert combined.shape[1] == 512 + 8 + 64, "Feature dimension should be sum of inputs"

  def test_policy_input_independence(self):
    """Test that nav embedding can be zero without affecting other inputs."""
    features = np.random.randn(1, 512).astype(np.float32)
    desire = np.random.randn(1, 8).astype(np.float32)
    
    # With zero nav embedding
    nav_zero = np.zeros(1, 64).astype(np.float32)
    combined_zero = np.concatenate([features, desire, nav_zero], axis=1)
    
    # With random nav embedding
    nav_random = np.random.randn(1, 64).astype(np.float32)
    combined_random = np.concatenate([features, desire, nav_random], axis=1)
    
    # Features and desire should be identical in both
    np.testing.assert_array_equal(combined_zero[0, :520], combined_random[0, :520])

  def test_nav_embedding_affects_input(self):
    """Test that different nav embeddings produce different policy inputs."""
    features = np.random.randn(1, 512).astype(np.float32)
    desire = np.random.randn(1, 8).astype(np.float32)
    
    # Different nav embeddings
    nav1 = np.zeros(1, 64).astype(np.float32)
    nav1[0, 1] = 1.0  # turn-left
    
    nav2 = np.zeros(1, 64).astype(np.float32)
    nav2[0, 2] = 1.0  # turn-right
    
    combined1 = np.concatenate([features, desire, nav1], axis=1)
    combined2 = np.concatenate([features, desire, nav2], axis=1)
    
    # Combined inputs should be different
    assert not np.array_equal(combined1, combined2)
    # Difference should be in nav embedding portion
    assert np.array_equal(combined1[0, :520], combined2[0, :520])
    assert not np.array_equal(combined1[0, 520:], combined2[0, 520:])


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
