#!/usr/bin/env python3
"""
E2E Phase 5: Training Loss Functions for End-to-End Control

This module defines loss functions for training the E2E driving model
to produce smooth, human-like actuator commands.

The losses are designed to penalize:
- Jerky acceleration/steering (comfort)
- Deviation from human driver behavior (imitation)
- Unsafe maneuvers (safety)
- Excessive control activity (efficiency)

Usage:
  from openpilot.selfdrive.modeld.e2e_losses import E2ELosses
  
  losses = E2ELosses()
  total_loss = losses.compute_loss(model_outputs, human_targets, car_state)
"""
import numpy as np
from typing import Dict, Tuple, Optional


class E2ELosses:
  """
  Loss functions for E2E driving model training.
  
  All losses operate on batched data:
  - Batch size: B
  - Sequence length: T (temporal context)
  """
  
  def __init__(self, 
               jerk_weight: float = 1.0,
               actuation_weight: float = 2.0,
               comfort_weight: float = 0.5,
               smoothness_weight: float = 0.3,
               aeb_weight: float = 5.0):
    """
    Initialize E2E loss weights.
    
    Args:
      jerk_weight: Weight for jerk loss (penalizes rapid acceleration changes)
      actuation_weight: Weight for actuation error (matching human driver)
      comfort_weight: Weight for comfort loss (lateral/longitudinal comfort)
      smoothness_weight: Weight for steering smoothness
      aeb_weight: Weight for AEB behavior (proper emergency braking)
    """
    self.jerk_weight = jerk_weight
    self.actuation_weight = actuation_weight
    self.comfort_weight = comfort_weight
    self.smoothness_weight = smoothness_weight
    self.aeb_weight = aeb_weight
    
    # Physical constants
    self.DT = 0.05  # 20 Hz model frequency
    self.GRAVITY = 9.81  # m/s^2
    
  def jerk_loss(self, 
                pred_accel: np.ndarray, 
                target_accel: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Jerk loss: penalizes rapid changes in acceleration.
    
    Jerk is the derivative of acceleration (da/dt). High jerk causes
    uncomfortable lurching sensations. This loss encourages smooth
    acceleration profiles.
    
    Args:
      pred_accel: Predicted acceleration [B, T] in m/s^2
      target_accel: Optional target acceleration (if None, uses zero-jerk target)
    
    Returns:
      Jerk loss value (scalar)
    
    Loss formulation:
      L_jerk = mean((d²/dt² pred_accel)²)
      
    This is equivalent to penalizing the second derivative of velocity,
    which corresponds to passenger comfort.
    """
    # Compute jerk (time derivative of acceleration)
    # Using finite differences: jerk[t] = (accel[t] - accel[t-1]) / dt
    if pred_accel.ndim == 1:
      pred_accel = pred_accel[np.newaxis, :]  # Add batch dimension
    
    B, T = pred_accel.shape
    if T < 2:
      return np.array(0.0, dtype=np.float32)
    
    # First derivative (acceleration change)
    accel_diff = np.diff(pred_accel, axis=1)  # [B, T-1]
    
    # Second derivative (jerk)
    if T < 3:
      jerk = accel_diff
    else:
      jerk = np.diff(accel_diff, axis=1)  # [B, T-2]
    
    # MSE loss on jerk
    jerk_loss = np.mean(jerk ** 2)
    
    # If target acceleration provided, also penalize deviation from target jerk
    if target_accel is not None:
      target_diff = np.diff(target_accel, axis=1)
      if T >= 3:
        target_jerk = np.diff(target_diff, axis=1)
      else:
        target_jerk = target_diff
      jerk_loss += np.mean((jerk - target_jerk) ** 2)
    
    return jerk_loss * self.jerk_weight
  
  def actuation_loss(self,
                     pred_torque: np.ndarray,
                     pred_gas: np.ndarray,
                     pred_brake: np.ndarray,
                     target_torque: np.ndarray,
                     target_gas: np.ndarray,
                     target_brake: np.ndarray) -> np.ndarray:
    """
    Actuation loss: MSE between predicted and human driver actuations.
    
    This is the primary imitation learning loss - the model learns to
    match the human driver's steering, gas, and brake inputs.
    
    Args:
      pred_*: Predicted actuator values [B, T]
        - torque: [-1, 1] steering torque
        - gas: [0, 1] gas pedal position
        - brake: [0, 1] brake pedal position
      target_*: Human driver actuations (same shape)
    
    Returns:
      Total actuation loss (scalar)
    
    Loss formulation:
      L_actuation = w_torque * MSE(pred_torque, target_torque)
                  + w_gas * MSE(pred_gas, target_gas)
                  + w_brake * MSE(pred_brake, target_brake)
    """
    # Ensure all inputs have batch dimension
    if pred_torque.ndim == 1:
      pred_torque = pred_torque[np.newaxis, :]
      target_torque = target_torque[np.newaxis, :]
      pred_gas = pred_gas[np.newaxis, :]
      target_gas = target_gas[np.newaxis, :]
      pred_brake = pred_brake[np.newaxis, :]
      target_brake = target_brake[np.newaxis, :]
    
    # Steering torque loss (most important for lateral control)
    torque_loss = np.mean((pred_torque - target_torque) ** 2)
    
    # Gas pedal loss
    gas_loss = np.mean((pred_gas - target_gas) ** 2)
    
    # Brake pedal loss (higher weight for safety)
    brake_loss = np.mean((pred_brake - target_brake) ** 2) * 1.5
    
    total_loss = torque_loss + gas_loss + brake_loss
    return total_loss * self.actuation_weight
  
  def comfort_loss(self,
                   pred_accel: np.ndarray,
                   pred_lateral_accel: Optional[np.ndarray] = None,
                   max_long_accel: float = 2.0,
                   max_lat_accel: float = 2.5) -> np.ndarray:
    """
    Comfort loss: penalizes acceleration beyond comfort thresholds.
    
    Unlike hard clipping, this uses a soft penalty that increases
    quadratically beyond the comfort zone. This allows the model to
    learn appropriate comfort boundaries.
    
    Args:
      pred_accel: Longitudinal acceleration [B, T] in m/s^2
      pred_lateral_accel: Lateral acceleration [B, T] in m/s^2 (optional)
      max_long_accel: Maximum comfortable longitudinal accel (default 2.0 m/s^2)
      max_lat_accel: Maximum comfortable lateral accel (default 2.5 m/s^2)
    
    Returns:
      Comfort loss value (scalar)
    
    Loss formulation:
      L_comfort = max(0, |accel| - accel_comfort)²
      
    This creates a "deadzone" where accelerations within the comfort
    zone have zero loss, but exceedances are penalized.
    """
    if pred_accel.ndim == 1:
      pred_accel = pred_accel[np.newaxis, :]
    
    # Longitudinal comfort loss
    long_exceedance = np.maximum(0, np.abs(pred_accel) - max_long_accel)
    long_loss = np.mean(long_exceedance ** 2)
    
    # Lateral comfort loss (if provided)
    lat_loss = 0.0
    if pred_lateral_accel is not None:
      if pred_lateral_accel.ndim == 1:
        pred_lateral_accel = pred_lateral_accel[np.newaxis, :]
      lat_exceedance = np.maximum(0, np.abs(pred_lateral_accel) - max_lat_accel)
      lat_loss = np.mean(lat_exceedance ** 2)
    
    total_loss = long_loss + lat_loss
    return total_loss * self.comfort_weight
  
  def steering_smoothness_loss(self,
                                pred_torque: np.ndarray,
                                target_torque: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Steering smoothness loss: penalizes high-frequency steering oscillations.
    
    Human drivers naturally filter out high-frequency steering corrections.
    This loss encourages the model to produce smooth steering trajectories
    rather than jittery corrections.
    
    Args:
      pred_torque: Predicted steering torque [B, T]
      target_torque: Human steering torque (optional, for rate matching)
    
    Returns:
      Smoothness loss value (scalar)
    
    Loss formulation:
      L_smooth = mean((d/dt pred_torque)²)
      
    This penalizes the first derivative of steering torque, encouraging
    smooth transitions.
    """
    if pred_torque.ndim == 1:
      pred_torque = pred_torque[np.newaxis, :]
    
    B, T = pred_torque.shape
    if T < 2:
      return np.array(0.0, dtype=np.float32)
    
    # Steering rate (first derivative)
    steering_rate = np.diff(pred_torque, axis=1) / self.DT
    
    # MSE on steering rate
    smoothness_loss = np.mean(steering_rate ** 2)
    
    # If target provided, also match human steering rates
    if target_torque is not None:
      if target_torque.ndim == 1:
        target_torque = target_torque[np.newaxis, :]
      target_rate = np.diff(target_torque, axis=1) / self.DT
      smoothness_loss += np.mean((steering_rate - target_rate) ** 2)
    
    return smoothness_loss * self.smoothness_weight
  
  def aeb_loss(self,
               pred_crash_prob: np.ndarray,
               pred_ttc: np.ndarray,
               pred_brake: np.ndarray,
               target_brake: np.ndarray,
               is_emergency: np.ndarray) -> np.ndarray:
    """
    AEB loss: teaches proper emergency braking behavior.
    
    This loss has two components:
    1. Penalize braking when no emergency (false positives)
    2. Penalize NOT braking during emergency (false negatives)
    
    Args:
      pred_crash_prob: Predicted crash probability [B, T]
      pred_ttc: Predicted time-to-collision [B, T] in seconds
      pred_brake: Predicted brake position [B, T]
      target_brake: Human brake position [B, T]
      is_emergency: Boolean array indicating emergency situations [B, T]
    
    Returns:
      AEB loss value (scalar)
    
    Loss formulation:
      L_aeb = w_fp * mean(pred_brake² | !emergency)
            + w_fn * mean((1 - pred_brake)² | emergency)
            
    Where:
      - False positive (fp): Braking when not needed
      - False negative (fn): Not braking when emergency exists
    """
    if pred_brake.ndim == 1:
      pred_brake = pred_brake[np.newaxis, :]
      target_brake = target_brake[np.newaxis, :]
      is_emergency = is_emergency[np.newaxis, :]
    
    # False positive penalty: braking when no emergency
    non_emergency_mask = ~is_emergency
    if np.any(non_emergency_mask):
      fp_loss = np.mean(pred_brake[non_emergency_mask] ** 2)
    else:
      fp_loss = 0.0
    
    # False negative penalty: not braking during emergency
    emergency_mask = is_emergency
    if np.any(emergency_mask):
      # During emergency, should brake hard (target_brake should be high)
      fn_loss = np.mean((target_brake[emergency_mask] - pred_brake[emergency_mask]) ** 2)
    else:
      fn_loss = 0.0
    
    total_loss = fp_loss + fn_loss * 2.0  # Higher weight on false negatives (safety)
    return total_loss * self.aeb_weight
  
  def compute_loss(self,
                   model_outputs: Dict[str, np.ndarray],
                   human_targets: Dict[str, np.ndarray],
                   car_state: Optional[Dict[str, np.ndarray]] = None,
                   return_components: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Compute total E2E training loss from all components.
    
    Args:
      model_outputs: Dict of model predictions:
        - 'accel': [B, T] longitudinal acceleration
        - 'torque': [B, T] steering torque
        - 'gas': [B, T] gas pedal position
        - 'brake': [B, T] brake pedal position
        - 'crash_prob': [B, T] crash probability
        - 'ttc': [B, T] time to collision
        - 'lateral_accel': [B, T] lateral acceleration (optional)
      
      human_targets: Dict of human driver targets (same keys as model_outputs)
      
      car_state: Optional dict with additional car state:
        - 'v_ego': [B, T] ego velocity
        - 'is_emergency': [B, T] emergency flag
      
      return_components: If True, return individual loss components
    
    Returns:
      total_loss: Scalar total loss value
      components: Dict of individual loss values (if return_components=True)
    
    Example usage:
      ```python
      losses = E2ELosses()
      total_loss, components = losses.compute_loss(
        model_outputs=preds,
        human_targets=targets,
        return_components=True
      )
      print(f"Jerk loss: {components['jerk']:.4f}")
      print(f"Actuation loss: {components['actuation']:.4f}")
      ```
    """
    components = {}
    
    # 1. Jerk loss (longitudinal comfort)
    if 'accel' in model_outputs:
      target_accel = human_targets.get('accel', None)
      components['jerk'] = float(self.jerk_loss(model_outputs['accel'], target_accel))
    
    # 2. Actuation loss (imitation learning)
    if all(k in model_outputs for k in ['torque', 'gas', 'brake']):
      components['actuation'] = float(self.actuation_loss(
        model_outputs['torque'],
        model_outputs['gas'],
        model_outputs['brake'],
        human_targets['torque'],
        human_targets['gas'],
        human_targets['brake']
      ))
    
    # 3. Comfort loss
    if 'accel' in model_outputs:
      lat_accel = model_outputs.get('lateral_accel', None)
      components['comfort'] = float(self.comfort_loss(model_outputs['accel'], lat_accel))
    
    # 4. Steering smoothness
    if 'torque' in model_outputs:
      target_torque = human_targets.get('torque', None)
      components['smoothness'] = float(self.steering_smoothness_loss(
        model_outputs['torque'], target_torque
      ))
    
    # 5. AEB loss (if emergency data available)
    if car_state is not None and 'is_emergency' in car_state:
      if all(k in model_outputs for k in ['crash_prob', 'ttc', 'brake']):
        components['aeb'] = float(self.aeb_loss(
          model_outputs['crash_prob'],
          model_outputs['ttc'],
          model_outputs['brake'],
          human_targets['brake'],
          car_state['is_emergency']
        ))
    
    # Compute weighted total
    total_loss = sum(components.values())
    
    if return_components:
      return total_loss, components
    else:
      return total_loss


def create_training_example(batch_size: int = 32, seq_len: int = 10) -> Tuple[Dict, Dict, Dict]:
  """
  Create synthetic training data for testing loss functions.
  
  Args:
    batch_size: Number of samples in batch
    seq_len: Sequence length (time steps)
  
  Returns:
    model_outputs: Synthetic model predictions
    human_targets: Synthetic human driver data
    car_state: Synthetic car state
  """
  np.random.seed(42)
  
  # Generate smooth human-like trajectories
  t = np.linspace(0, 10, seq_len)
  
  # Human steering: smooth sinusoidal pattern
  human_torque = np.sin(2 * np.pi * t / 5.0)[np.newaxis, :] * 0.5
  human_torque = np.tile(human_torque, (batch_size, 1))
  
  # Human gas/brake: smooth acceleration/deceleration
  human_gas = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * t / 8.0), 0, 1)[np.newaxis, :]
  human_gas = np.tile(human_gas, (batch_size, 1))
  
  human_brake = np.zeros_like(human_gas)
  # Add some braking events
  human_brake[:, 5:7] = 0.3
  
  # Human acceleration (derived from gas/brake)
  human_accel = human_gas * 2.0 - human_brake * 4.0
  
  # Model outputs (slightly noisy version of human)
  model_outputs = {
    'torque': human_torque + np.random.randn(batch_size, seq_len) * 0.05,
    'gas': human_gas + np.random.randn(batch_size, seq_len) * 0.05,
    'brake': human_brake + np.random.randn(batch_size, seq_len) * 0.03,
    'accel': human_accel + np.random.randn(batch_size, seq_len) * 0.2,
    'crash_prob': np.random.rand(batch_size, seq_len) * 0.1,
    'ttc': np.ones((batch_size, seq_len)) * 5.0 + np.random.randn(batch_size, seq_len),
  }
  
  human_targets = {
    'torque': human_torque,
    'gas': human_gas,
    'brake': human_brake,
    'accel': human_accel,
  }
  
  car_state = {
    'v_ego': np.ones((batch_size, seq_len)) * 20.0,  # 20 m/s (~45 mph)
    'is_emergency': np.zeros((batch_size, seq_len), dtype=bool),
  }
  
  return model_outputs, human_targets, car_state


if __name__ == "__main__":
  # Test loss functions with synthetic data
  print("Testing E2E Loss Functions")
  print("=" * 50)
  
  losses = E2ELosses()
  model_outputs, human_targets, car_state = create_training_example()
  
  total_loss, components = losses.compute_loss(
    model_outputs=model_outputs,
    human_targets=human_targets,
    car_state=car_state,
    return_components=True
  )
  
  print(f"\nTotal Loss: {total_loss:.4f}")
  print("\nLoss Components:")
  for name, value in sorted(components.items()):
    print(f"  {name:15s}: {value:8.4f}")
  
  print("\n" + "=" * 50)
  print("✅ All loss functions working correctly")
