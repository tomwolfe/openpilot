#!/usr/bin/env python3
"""
E2E Phase 2: Vehicle Conditioning Module

This module extracts vehicle parameters and creates an embedding vector
that is fed into the neural network, allowing the model to learn
vehicle-specific dynamics without manual tuning.

The model learns how much torque/acceleration is appropriate for:
- A 3,000 lb sedan vs. a 6,000 lb truck
- Different wheelbase lengths
- Different steering ratios
"""
import numpy as np
from cereal import car


class VehicleConditioning:
  """
  Extracts and normalizes vehicle parameters for model input.
  
  The embedding vector is concatenated with the model's feature buffer
  and passed through the policy network.
  """
  
  # Normalization constants for vehicle parameters
  # These values represent typical ranges for passenger vehicles
  MASS_NORM = 2000.0        # kg: normalize around 2000 kg (4400 lbs)
  WHEELBASE_NORM = 2.8      # m: normalize around 2.8 m (110 inches)
  STEER_RATIO_NORM = 15.0   # : normalize around 15:1 ratio
  CENTER_TO_FRONT_NORM = 1.5  # m: normalize around 1.5 m
  
  def __init__(self, CP: car.CarParams):
    """
    Initialize vehicle conditioning from CarParams.
    
    Args:
      CP: CarParams message containing vehicle configuration
    """
    self.CP = CP
    
    # Extract and normalize vehicle parameters
    self.vehicle_embedding = self._create_embedding()
    
  def _create_embedding(self) -> np.ndarray:
    """
    Create normalized vehicle embedding vector.
    
    Returns:
      Normalized embedding vector [mass, wheelbase, steer_ratio, center_to_front]
    """
    # Extract parameters
    mass = getattr(self.CP, 'mass', 2000.0)
    wheelbase = getattr(self.CP, 'wheelbase', 2.8)
    steer_ratio = getattr(self.CP, 'steerRatio', 15.0)
    center_to_front = getattr(self.CP, 'centerToFront', 1.5)
    
    # Normalize to [0, 1] range for neural network input
    embedding = np.array([
      mass / self.MASS_NORM,
      wheelbase / self.WHEELBASE_NORM,
      steer_ratio / self.STEER_RATIO_NORM,
      center_to_front / self.CENTER_TO_FRONT_NORM,
    ], dtype=np.float32)
    
    # Clip to reasonable bounds
    embedding = np.clip(embedding, 0.0, 2.0)
    
    return embedding
  
  def get_embedding(self) -> np.ndarray:
    """Get the current vehicle embedding vector."""
    return self.vehicle_embedding.copy()
  
  def update(self, CP: car.CarParams = None):
    """
    Update vehicle parameters if they change.
    
    Args:
      CP: Optional new CarParams (if None, uses existing)
    """
    if CP is not None:
      self.CP = CP
      self.vehicle_embedding = self._create_embedding()


def get_vehicle_embedding_shape() -> tuple:
  """
  Returns the expected shape of the vehicle embedding vector.
  
  Returns:
    Tuple (batch_size, embedding_dim) = (1, 4)
  """
  return (1, 4)


def integrate_with_modeld():
  """
  Integration guide for adding vehicle conditioning to modeld.
  
  This function documents the steps needed to integrate vehicle
  conditioning into the modeld pipeline.
  
  Steps:
  1. In modeld.py, create VehicleConditioning instance:
     ```
     from openpilot.selfdrive.modeld.vehicle_conditioning import VehicleConditioning
     VC = VehicleConditioning(CP)
     ```
  
  2. Add vehicle_embedding to model inputs:
     ```
     inputs['vehicle_embedding'] = VC.get_embedding()
     ```
  
  3. Update model input shapes in ModelState:
     ```
     self.policy_input_shapes['vehicle_embedding'] = (1, 4)
     ```
  
  4. Modify model architecture to accept vehicle embedding:
     - Concatenate with feature buffer before final layers
     - Or use as conditioning input to FiLM layers
  
  5. Train model with vehicle parameters as input
  """
  pass
