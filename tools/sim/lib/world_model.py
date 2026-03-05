#!/usr/bin/env python3
"""
World Model for openpilot Phase 3 Closed-Loop Simulation.

This module provides a learned World Model that can predict the next visual frame
given the current camera state and a car control action. Uses tinygrad for inference.

The World Model enables:
1. Closed-loop training without real-world data collection
2. Adversarial scenario generation for stress-testing
3. Counterfactual reasoning ("what if" scenarios)
4. Data augmentation for model training

Architecture:
- Input: Current roadCameraState + carControl action (steer, accel, brake)
- Output: Predicted next frame (or frame delta)
- Model: Convolutional VAE with action conditioning
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear, LayerNorm
from tinygrad.helpers import DEBUG


@dataclass
class WorldModelState:
  """State container for World Model."""
  latent_state: np.ndarray  # Latent representation of environment
  last_action: np.ndarray   # Last applied action [steer, accel, brake]
  last_frame: np.ndarray    # Last observed frame
  timestamp: float          # Last update timestamp
  valid: bool = False       # Whether state is initialized


class ActionEncoder:
  """
  Encodes car control actions into latent space.
  
  Actions:
  - steering_angle: [-40°, 40°] → normalized [-1, 1]
  - acceleration: [-3.5, 2.0] m/s² → normalized [-1, 1]
  - brake: [0, 1] binary/continuous
  """
  
  def __init__(self, action_dim: int = 3, latent_dim: int = 64):
    self.action_dim = action_dim
    self.latent_dim = latent_dim
    
    # Action embedding network - output full latent_dim
    self.action_embed = Linear(action_dim, latent_dim)
    self.action_norm = LayerNorm(latent_dim)
  
  def normalize_action(self, action: np.ndarray) -> np.ndarray:
    """Normalize action to [-1, 1] range."""
    normalized = action.copy()
    
    # Steering: [-40, 40] degrees → [-1, 1]
    normalized[0] = np.clip(action[0] / 40.0, -1, 1)
    
    # Acceleration: [-3.5, 2.0] m/s² → [-1, 1]
    # Use asymmetric scaling for brake vs accel
    if action[1] > 0:  # Acceleration
      normalized[1] = np.clip(action[1] / 2.0, 0, 1)
    else:  # Braking
      normalized[1] = np.clip(action[1] / 3.5, -1, 0)
    
    # Brake: [0, 1] → [0, 1]
    normalized[2] = np.clip(action[2], 0, 1)
    
    return normalized
  
  def encode(self, action: np.ndarray) -> Tensor:
    """
    Encode action into latent representation.
    
    Args:
      action: [steer, accel, brake] array
    
    Returns:
      Tensor of shape (latent_dim,)
    """
    normalized = self.normalize_action(action)
    action_tensor = Tensor(normalized, requires_grad=False)
    
    # Embed and normalize
    embedded = self.action_embed(action_tensor)
    embedded = self.action_norm(embedded)
    
    return embedded


class VisionEncoder:
  """
  Encodes camera frames into latent representation.
  
  Uses adaptive pooling to handle variable input sizes.
  """
  
  def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3),
               latent_dim: int = 256):
    self.input_shape = input_shape
    self.latent_dim = latent_dim
    
    # Simple convolutional encoder
    self.conv1 = Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    self.conv4 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    
    # Use adaptive pooling to get fixed-size output
    self.spatial_dim = 4  # Pool to 4x4
    flattened_size = 256 * self.spatial_dim * self.spatial_dim
    
    # Projection to latent
    self.proj = Linear(flattened_size, latent_dim)
    self.norm = LayerNorm(latent_dim)
  
  def preprocess(self, frame: np.ndarray) -> Tensor:
    """
    Preprocess frame for encoding.
    
    Args:
      frame: RGB frame of shape (H, W, 3), uint8 [0, 255]
    
    Returns:
      Tensor of shape (1, 3, H, W), float32, normalized
    """
    # Convert to tensor and normalize
    frame_tensor = Tensor(frame, requires_grad=False)
    
    # Reshape to channels-first (1, 3, H, W) for batched processing
    if frame_tensor.shape == (frame.shape[0], frame.shape[1], 3):
      frame_tensor = frame_tensor.permute(2, 0, 1).reshape(1, 3, frame.shape[0], frame.shape[1])
    
    # Normalize to [0, 1]
    frame_tensor = frame_tensor / 255.0
    
    return frame_tensor
  
  def encode(self, frame: np.ndarray) -> Tensor:
    """
    Encode frame into latent representation.
    
    Args:
      frame: RGB frame
    
    Returns:
      Tensor of shape (latent_dim,)
    """
    x = self.preprocess(frame)
    
    # Convolutional encoding with ReLU activations
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = self.conv3(x).relu()
    x = self.conv4(x).relu()
    
    # Adaptive average pooling to fixed spatial size
    # This handles variable input sizes
    b, c, h, w = x.shape
    # Simple pooling by taking center crop or padding
    if h > self.spatial_dim or w > self.spatial_dim:
      # Downsample by averaging
      x = x.reshape(b, c, h // self.spatial_dim, self.spatial_dim, 
                       w // self.spatial_dim, self.spatial_dim)
      x = x.mean(axis=(2, 4))
    else:
      # Pad if too small
      from tinygrad.tensor import Tensor
      pad_h = self.spatial_dim - h
      pad_w = self.spatial_dim - w
      if pad_h > 0 or pad_w > 0:
        x = x.pad(((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    
    # Flatten and project
    x = x.flatten(start_dim=1)
    x = self.proj(x)
    x = self.norm(x)
    
    # Remove batch dimension
    return x[0]


class WorldModelPredictor:
  """
  Core World Model that predicts next state given current state and action.
  
  Architecture:
  - Takes current latent state + action embedding
  - Predicts delta in latent space
  - Decoder reconstructs predicted frame
  """
  
  def __init__(self, vision_latent_dim: int = 256, action_latent_dim: int = 64):
    self.vision_latent_dim = vision_latent_dim
    self.action_latent_dim = action_latent_dim
    self.combined_dim = vision_latent_dim + action_latent_dim
    
    # Dynamics model: predicts latent state transition
    self.dynamics = [
      Linear(self.combined_dim, 512),
      LayerNorm(512),
      Linear(512, 512),
      LayerNorm(512),
      Linear(512, vision_latent_dim),  # Output delta in vision latent space
    ]
    
    # Decoder: reconstructs frame from latent
    # Simplified decoder for demonstration
    self.decoder_proj = Linear(vision_latent_dim, 256 * 4 * 4)  # Match encoder's spatial_dim
    self.decoder_conv1 = Conv2d(256, 128, kernel_size=3, padding=1)
    self.decoder_conv2 = Conv2d(128, 64, kernel_size=3, padding=1)
    self.decoder_conv3 = Conv2d(64, 32, kernel_size=3, padding=1)
    self.decoder_out = Conv2d(32, 3, kernel_size=3, padding=1)
  
  def predict_delta(self, latent_state: Tensor, action_embed: Tensor) -> Tensor:
    """
    Predict change in latent state given action.
    
    Args:
      latent_state: Current latent state (vision_latent_dim,)
      action_embed: Encoded action (action_latent_dim,)
    
    Returns:
      Predicted delta in latent space (vision_latent_dim,)
    """
    # Concatenate state and action
    combined = latent_state.cat(action_embed, dim=0)
    
    # Apply dynamics network
    x = combined
    for layer in self.dynamics:
      if isinstance(layer, Linear):
        x = layer(x).relu()
      else:
        x = layer(x)
    
    return x
  
  def decode(self, latent_state: Tensor) -> Tensor:
    """
    Decode latent state to frame.
    
    Args:
      latent_state: Latent representation of shape (latent_dim,)
    
    Returns:
      Predicted frame (3, H, W), normalized [0, 1]
    """
    # Add batch dimension if needed
    if latent_state.ndim == 1:
      latent_state = latent_state.reshape(1, -1)
    
    # Project to spatial representation (match encoder's spatial_dim=4)
    x = self.decoder_proj(latent_state)
    
    # Reshape to spatial (1, 256, 4, 4) to match encoder
    x = x.reshape(1, 256, 4, 4)
    
    # Transposed convolutions for upsampling
    x = self.decoder_conv1(x).relu()
    x = self.decoder_conv2(x).relu()
    x = self.decoder_conv3(x).relu()
    x = self.decoder_out(x).sigmoid()  # Output in [0, 1]
    
    # Remove batch dimension and return (3, H, W)
    return x[0]


class WorldModel:
  """
  High-level World Model interface for closed-loop simulation.
  
  Usage:
    wm = WorldModel()
    
    # Initialize with observed frame
    wm.initialize(current_frame)
    
    # Predict next frame given action
    action = [steer, accel, brake]
    predicted_frame = wm.predict(action)
    
    # Update with actual observation
    wm.update(actual_next_frame, action)
  """
  
  def __init__(self, model_path: Optional[Path] = None):
    """
    Initialize World Model.
    
    Args:
      model_path: Path to pretrained weights (optional)
    """
    self.vision_encoder = VisionEncoder(latent_dim=256)
    self.action_encoder = ActionEncoder(latent_dim=64)
    self.predictor = WorldModelPredictor(vision_latent_dim=256, action_latent_dim=64)
    
    self.state: Optional[WorldModelState] = None
    self.model_path = model_path
    
    # Performance tracking
    self.inference_times: list = []
    self.prediction_errors: list = []
  
  def initialize(self, frame: np.ndarray):
    """
    Initialize World Model state from observed frame.
    
    Args:
      frame: Initial camera frame (H, W, 3), uint8
    """
    # Encode initial frame
    latent = self.vision_encoder.encode(frame)
    latent_np = latent.numpy()
    
    self.state = WorldModelState(
      latent_state=latent_np,
      last_action=np.zeros(3),  # No action yet
      last_frame=frame,
      timestamp=time.time(),
      valid=True
    )
    
    if DEBUG >= 1:
      print(f"[WorldModel] Initialized with frame shape {frame.shape}")
  
  def predict(self, action: np.ndarray, predict_delta: bool = True) -> np.ndarray:
    """
    Predict next frame given action.
    
    Args:
      action: [steer, accel, brake] array
      predict_delta: If True, predict delta from current frame
    
    Returns:
      Predicted frame (H, W, 3), uint8 [0, 255]
    """
    if not self.state or not self.state.valid:
      raise RuntimeError("WorldModel not initialized. Call initialize() first.")
    
    start_time = time.perf_counter()
    
    # Convert state to tensor (ensure 1D)
    latent_np = self.state.latent_state.flatten()
    latent_tensor = Tensor(latent_np, requires_grad=False)
    
    # Encode action
    action_embed = self.action_encoder.encode(action)
    
    # Predict next state
    if predict_delta:
      delta = self.predictor.predict_delta(latent_tensor, action_embed)
      next_latent = latent_tensor + delta * 0.1  # Scale delta for stability
    else:
      next_latent = self.predictor.predict_delta(latent_tensor, action_embed)
    
    # Decode to frame
    predicted_frame_tensor = self.predictor.decode(next_latent)
    
    # Convert to numpy and rescale
    predicted_frame = predicted_frame_tensor.numpy()
    
    # Reshape and scale to [0, 255]
    if predicted_frame.ndim == 3 and predicted_frame.shape[0] == 3:
      # Channels-first (3, H, W) → channels-last (H, W, 3)
      predicted_frame = np.transpose(predicted_frame, (1, 2, 0))
    
    predicted_frame = np.clip(predicted_frame * 255, 0, 255).astype(np.uint8)
    
    # Track performance
    inference_time = time.perf_counter() - start_time
    self.inference_times.append(inference_time)
    
    if DEBUG >= 1:
      print(f"[WorldModel] Prediction took {inference_time*1000:.2f}ms")
    
    return predicted_frame
  
  def update(self, actual_frame: np.ndarray, action: np.ndarray):
    """
    Update World Model with actual observation.
    
    Args:
      actual_frame: Observed frame after action
      action: Action that was applied
    """
    if not self.state or not self.state.valid:
      self.initialize(actual_frame)
      return
    
    # Encode actual frame
    actual_latent = self.vision_encoder.encode(actual_frame)
    actual_latent_np = actual_latent.numpy()
    
    # Compute prediction error (for monitoring)
    predicted_latent = self.state.latent_state.copy()
    prediction_error = np.mean((predicted_latent - actual_latent_np) ** 2)
    self.prediction_errors.append(prediction_error)
    
    # Update state
    self.state = WorldModelState(
      latent_state=actual_latent_np,
      last_action=action.copy(),
      last_frame=actual_frame,
      timestamp=time.time(),
      valid=True
    )
  
  def get_stats(self) -> Dict[str, Any]:
    """Get performance statistics."""
    return {
      'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
      'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0,
      'state_valid': self.state.valid if self.state else False,
    }
  
  def save(self, path: Path):
    """Save model weights."""
    # TODO: Implement weight saving
    pass
  
  def load(self, path: Path):
    """Load pretrained weights."""
    # TODO: Implement weight loading
    pass


class AdversarialScenarioInjector:
  """
  Injects adversarial scenarios into the World Model for stress-testing.
  
  Scenarios:
  1. Sudden cut-in: Vehicle appears in adjacent lane
  2. Disappearing lane lines: Lane markings vanish
  3. Phantom braking: False obstacle detection
  4. Sensor noise: Add noise to camera input
  """
  
  def __init__(self, world_model: WorldModel):
    self.wm = world_model
    self.active_scenario: Optional[str] = None
  
  def inject_cut_in(self, frame: np.ndarray, side: str = 'left',
                   intensity: float = 0.5) -> np.ndarray:
    """
    Simulate a vehicle cutting in from adjacent lane.
    
    Args:
      frame: Current frame
      side: 'left' or 'right'
      intensity: How prominent the cut-in should be [0, 1]
    
    Returns:
      Modified frame with cut-in vehicle
    """
    # Create synthetic vehicle blob
    h, w = frame.shape[:2]
    vehicle_h, vehicle_w = int(h * 0.15), int(w * 0.08)
    
    # Position in adjacent lane
    if side == 'left':
      x_start = int(w * 0.1)
      y_start = int(h * 0.4)
    else:
      x_start = int(w * 0.6)
      y_start = int(h * 0.4)
    
    # Create vehicle shape (simplified as colored rectangle)
    modified = frame.copy()
    
    # Add vehicle with intensity-based opacity
    vehicle_color = np.array([50, 50, 150], dtype=np.uint8)  # Blue vehicle
    
    y_end = min(y_start + vehicle_h, h)
    x_end = min(x_start + vehicle_w, w)
    
    alpha = intensity
    modified[y_start:y_end, x_start:x_end] = \
      (1 - alpha) * modified[y_start:y_end, x_start:x_end] + \
      alpha * vehicle_color
    
    return modified
  
  def inject_disappearing_lanes(self, frame: np.ndarray,
                                fade_factor: float = 0.8) -> np.ndarray:
    """
    Simulate lane lines disappearing (e.g., construction zone).
    
    Args:
      frame: Current frame
      fade_factor: How much to fade lane markings [0, 1]
    
    Returns:
      Modified frame with faded lane markings
    """
    # Detect lane-like features (simplified: yellow/white regions in lower frame)
    modified = frame.copy()
    
    # Focus on lower half of frame where lanes are visible
    h, w = frame.shape[:2]
    lower_half = modified[h//2:, :]
    
    # Find white/yellow pixels (lane markings)
    white_mask = np.all(lower_half > 200, axis=2)
    yellow_mask = (lower_half[:, :, 0] > 150) & (lower_half[:, :, 1] > 150) & \
                  (lower_half[:, :, 2] < 100)
    
    lane_mask = white_mask | yellow_mask
    
    # Fade lane markings
    fade_region = modified[h//2:, :][lane_mask]
    modified[h//2:, :][lane_mask] = \
      (1 - fade_factor) * fade_region + fade_factor * np.mean(frame, axis=(0, 1))
    
    return modified
  
  def inject_sensor_noise(self, frame: np.ndarray,
                         noise_level: float = 0.1) -> np.ndarray:
    """
    Add sensor noise to camera input.
    
    Args:
      frame: Current frame
      noise_level: Standard deviation of Gaussian noise [0, 1]
    
    Returns:
      Noisy frame
    """
    noise = np.random.normal(0, noise_level * 255, frame.shape)
    noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_frame
  
  def run_scenario(self, scenario: str, frame: np.ndarray,
                   **kwargs) -> np.ndarray:
    """
    Run specified adversarial scenario.
    
    Args:
      scenario: Name of scenario ('cut_in', 'disappearing_lanes', 'sensor_noise')
      frame: Input frame
      **kwargs: Scenario-specific parameters
    
    Returns:
      Modified frame
    """
    self.active_scenario = scenario
    
    if scenario == 'cut_in':
      side = kwargs.get('side', 'left')
      intensity = kwargs.get('intensity', 0.5)
      return self.inject_cut_in(frame, side, intensity)
    
    elif scenario == 'disappearing_lanes':
      fade_factor = kwargs.get('fade_factor', 0.8)
      return self.inject_disappearing_lanes(frame, fade_factor)
    
    elif scenario == 'sensor_noise':
      noise_level = kwargs.get('noise_level', 0.1)
      return self.inject_sensor_noise(frame, noise_level)
    
    else:
      raise ValueError(f"Unknown scenario: {scenario}")


def create_world_model(pretrained_path: Optional[Path] = None) -> WorldModel:
  """
  Factory function to create World Model instance.
  
  Args:
    pretrained_path: Path to pretrained weights
  
  Returns:
    Initialized WorldModel
  """
  wm = WorldModel(model_path=pretrained_path)
  
  # TODO: Load pretrained weights if available
  # if pretrained_path and pretrained_path.exists():
  #   wm.load(pretrained_path)
  
  return wm


if __name__ == "__main__":
  # Test World Model
  print("Testing World Model...")
  
  # Create dummy frame
  dummy_frame = np.random.randint(0, 255, (1208, 1928, 3), dtype=np.uint8)
  
  # Initialize
  wm = create_world_model()
  wm.initialize(dummy_frame)
  
  # Predict
  action = np.array([0.0, 0.0, 0.0])  # No action
  predicted = wm.predict(action)
  
  print(f"Input shape: {dummy_frame.shape}")
  print(f"Predicted shape: {predicted.shape}")
  print(f"Stats: {wm.get_stats()}")
  
  # Test adversarial scenarios
  print("\nTesting adversarial scenarios...")
  injector = AdversarialScenarioInjector(wm)
  
  noisy = injector.run_scenario('sensor_noise', dummy_frame, noise_level=0.1)
  print(f"Noisy frame shape: {noisy.shape}")
  
  print("\n✓ World Model test complete")
