#!/usr/bin/env python3
"""
E2E Metrics Logging Module

This module provides comprehensive metrics collection and logging for
the E2E driving system. It tracks:
- Control quality (tracking error, smoothness)
- Comfort metrics (jerk, acceleration exceedances)
- Safety metrics (AEB activations, LDW warnings)
- Model confidence and health

Usage:
  from openpilot.selfdrive.controls.lib.e2e_metrics import E2EMetrics
  
  metrics = E2EMetrics()
  metrics.update(model_outputs, car_state, actuators)
  
  # Get summary for logging
  summary = metrics.get_summary()
"""
import numpy as np
from collections import deque
from typing import Dict, Optional, Any

# Use DT_CTRL from realtime if available, otherwise default
try:
  from openpilot.common.realtime import DT_CTRL
except ImportError:
  DT_CTRL = 0.01  # 100Hz default


class E2EMetrics:
  """
  Collects and computes E2E driving metrics.
  
  All metrics are computed over a sliding window to provide
  both instantaneous and statistical measures.
  """
  
  def __init__(self, window_size: int = 100):
    """
    Initialize metrics collector.
    
    Args:
      window_size: Number of samples for sliding window (default 100 = 1 second at 100Hz)
    """
    self.window_size = window_size
    
    # Sliding windows for metrics
    self.accel_history = deque(maxlen=window_size)
    self.torque_history = deque(maxlen=window_size)
    self.jerk_history = deque(maxlen=window_size)
    self.steer_rate_history = deque(maxlen=window_size)
    
    # Event counters
    self.aeb_activations = 0
    self.ldw_warnings = 0
    self.comfort_violations = 0
    self.saturation_events = 0
    
    # Model health tracking
    self.model_output_std = deque(maxlen=window_size)
    self.prediction_confidence = deque(maxlen=window_size)
    
    # State tracking
    self.prev_accel = 0.0
    self.prev_torque = 0.0
    self.prev_torque_time = 0.0
    
    # Comfort thresholds
    self.MAX_COMFORT_ACCEL = 2.0  # m/s²
    self.MAX_COMFORT_JERK = 2.0  # m/s³
    self.MAX_STEER_RATE = 100.0  # deg/s
    
  def update(self, 
             model_outputs: Dict[str, float],
             car_state: Dict[str, Any],
             actuators: Dict[str, float],
             timestamp: Optional[float] = None):
    """
    Update metrics with new control cycle data.
    
    Args:
      model_outputs: Model predictions (accel, torque, etc.)
      car_state: Current vehicle state (vEgo, steeringTorque, etc.)
      actuators: Actual actuator commands sent
      timestamp: Optional timestamp (uses counter if not provided)
    """
    # Extract values
    accel = actuators.get('accel', 0.0)
    torque = actuators.get('torque', 0.0)
    
    # Compute derived metrics
    jerk = (accel - self.prev_accel) / DT_CTRL
    steer_rate = (torque - self.prev_torque) / DT_CTRL * 100  # Convert to deg/s approx
    
    # Update history buffers
    self.accel_history.append(accel)
    self.torque_history.append(torque)
    self.jerk_history.append(jerk)
    self.steer_rate_history.append(steer_rate)
    
    # Track model output statistics
    if 'crash_prob' in model_outputs:
      self.prediction_confidence.append(1.0 - model_outputs['crash_prob'])
    
    # Count events
    self._count_events(model_outputs, car_state, accel, jerk, steer_rate)
    
    # Update state
    self.prev_accel = accel
    self.prev_torque = torque
    self.prev_torque_time = timestamp or self.prev_torque_time + DT_CTRL
    
  def _count_events(self, 
                    model_outputs: Dict[str, float],
                    car_state: Dict[str, Any],
                    accel: float,
                    jerk: float,
                    steer_rate: float):
    """Count discrete events for metrics."""
    
    # AEB activations
    if model_outputs.get('aeb_imminent', False):
      self.aeb_activations += 1
    
    # LDW warnings
    if model_outputs.get('ldw_warning', False):
      self.ldw_warnings += 1
    
    # Comfort violations
    if abs(accel) > self.MAX_COMFORT_ACCEL:
      self.comfort_violations += 1
    if abs(jerk) > self.MAX_COMFORT_JERK:
      self.comfort_violations += 1
    if abs(steer_rate) > self.MAX_STEER_RATE:
      self.steer_rate_history.append(steer_rate)
    
  def get_summary(self) -> Dict[str, float]:
    """
    Get summary statistics for all metrics.
    
    Returns:
      Dict with metric names and values
    """
    summary = {}
    
    # Acceleration statistics
    if len(self.accel_history) > 0:
      accel_arr = np.array(self.accel_history)
      summary['accel_mean'] = float(np.mean(accel_arr))
      summary['accel_std'] = float(np.std(accel_arr))
      summary['accel_max'] = float(np.max(np.abs(accel_arr)))
      summary['accel_p95'] = float(np.percentile(np.abs(accel_arr), 95))
    
    # Jerk statistics
    if len(self.jerk_history) > 0:
      jerk_arr = np.array(self.jerk_history)
      summary['jerk_mean'] = float(np.mean(np.abs(jerk_arr)))
      summary['jerk_std'] = float(np.std(jerk_arr))
      summary['jerk_max'] = float(np.max(np.abs(jerk_arr)))
      summary['jerk_p95'] = float(np.percentile(np.abs(jerk_arr), 95))
    
    # Steering statistics
    if len(self.torque_history) > 0:
      torque_arr = np.array(self.torque_history)
      summary['torque_mean'] = float(np.mean(torque_arr))
      summary['torque_std'] = float(np.std(torque_arr))
      summary['torque_max'] = float(np.max(np.abs(torque_arr)))
    
    # Steering rate statistics
    if len(self.steer_rate_history) > 0:
      steer_rate_arr = np.array(self.steer_rate_history)
      summary['steer_rate_mean'] = float(np.mean(np.abs(steer_rate_arr)))
      summary['steer_rate_max'] = float(np.max(np.abs(steer_rate_arr)))
    
    # Model confidence
    if len(self.prediction_confidence) > 0:
      summary['model_confidence_mean'] = float(np.mean(self.prediction_confidence))
      summary['model_confidence_min'] = float(np.min(self.prediction_confidence))
    
    # Event rates (per minute, assuming 100Hz)
    minutes = len(self.accel_history) * DT_CTRL / 60.0
    if minutes > 0:
      summary['aeb_rate_per_min'] = self.aeb_activations / minutes
      summary['ldw_rate_per_min'] = self.ldw_warnings / minutes
      summary['comfort_violations_per_min'] = self.comfort_violations / minutes
    
    # Total counts
    summary['aeb_total'] = self.aeb_activations
    summary['ldw_total'] = self.ldw_warnings
    summary['comfort_violations_total'] = self.comfort_violations
    
    # Quality scores (0-100, higher is better)
    summary['comfort_score'] = self._compute_comfort_score()
    summary['smoothness_score'] = self._compute_smoothness_score()
    
    return summary
  
  def _compute_comfort_score(self) -> float:
    """
    Compute overall comfort score (0-100).
    
    Based on:
    - Acceleration exceedances
    - Jerk levels
    - Steering smoothness
    
    Returns:
      Score from 0 (very uncomfortable) to 100 (perfect comfort)
    """
    if len(self.accel_history) == 0:
      return 100.0
    
    accel_arr = np.array(self.accel_history)
    jerk_arr = np.array(self.jerk_history)
    
    # Penalty for acceleration exceedances
    accel_exceedance = np.mean(np.maximum(0, np.abs(accel_arr) - self.MAX_COMFORT_ACCEL))
    accel_penalty = min(50, accel_exceedance * 25)
    
    # Penalty for high jerk
    jerk_exceedance = np.mean(np.maximum(0, np.abs(jerk_arr) - self.MAX_COMFORT_JERK))
    jerk_penalty = min(50, jerk_exceedance * 25)
    
    score = 100.0 - accel_penalty - jerk_penalty
    return max(0.0, min(100.0, score))
  
  def _compute_smoothness_score(self) -> float:
    """
    Compute steering smoothness score (0-100).
    
    Based on:
    - Steering rate
    - Torque variability
    
    Returns:
      Score from 0 (jerky) to 100 (smooth)
    """
    if len(self.torque_history) == 0:
      return 100.0
    
    torque_arr = np.array(self.torque_history)
    steer_rate_arr = np.array(self.steer_rate_history)
    
    # Penalty for high steering rates
    rate_exceedance = np.mean(np.maximum(0, np.abs(steer_rate_arr) - self.MAX_STEER_RATE))
    rate_penalty = min(50, rate_exceedance * 0.5)
    
    # Penalty for torque variability
    torque_std = np.std(torque_arr)
    std_penalty = min(50, torque_std * 100)
    
    score = 100.0 - rate_penalty - std_penalty
    return max(0.0, min(100.0, score))
  
  def reset(self):
    """Reset all metrics and counters."""
    self.accel_history.clear()
    self.torque_history.clear()
    self.jerk_history.clear()
    self.steer_rate_history.clear()
    self.model_output_std.clear()
    self.prediction_confidence.clear()
    
    self.aeb_activations = 0
    self.ldw_warnings = 0
    self.comfort_violations = 0
    self.saturation_events = 0
    
    self.prev_accel = 0.0
    self.prev_torque = 0.0
    self.prev_torque_time = 0.0
  
  def get_alerts(self) -> list:
    """
    Get list of active alerts based on metrics.
    
    Returns:
      List of alert strings
    """
    alerts = []
    
    # Check for excessive jerk
    if len(self.jerk_history) > 10:
      recent_jerk = np.array(list(self.jerk_history)[-10:])
      if np.mean(np.abs(recent_jerk)) > self.MAX_COMFORT_JERK * 1.5:
        alerts.append("High jerk detected - model may need retraining")
    
    # Check for frequent AEB
    if len(self.accel_history) > 100:
      minutes = len(self.accel_history) * DT_CTRL / 60.0
      if minutes > 0 and self.aeb_activations / minutes > 1.0:
        alerts.append("Frequent AEB activations - check model crash prediction")
    
    # Check for low model confidence
    if len(self.prediction_confidence) > 10:
      recent_conf = np.array(list(self.prediction_confidence)[-10:])
      if np.mean(recent_conf) < 0.7:
        alerts.append("Low model confidence - consider disengaging")
    
    return alerts


def create_metrics_log_entry(metrics: E2EMetrics) -> Dict[str, float]:
  """
  Create a log entry from metrics for storage.
  
  Args:
    metrics: E2EMetrics instance
    
  Returns:
    Dict suitable for logging/storage
  """
  summary = metrics.get_summary()
  
  # Add timestamp and metadata
  import time
  entry = {
    'timestamp': time.time(),
    **summary
  }
  
  # Add alerts
  entry['alerts'] = metrics.get_alerts()
  
  return entry


if __name__ == "__main__":
  # Test metrics collection
  print("Testing E2E Metrics Collection")
  print("=" * 50)
  
  metrics = E2EMetrics(window_size=100)
  
  # Simulate 100 control cycles
  for i in range(100):
    model_outputs = {
      'accel_pred': np.sin(i * 0.1) * 1.0,
      'torque_pred': np.cos(i * 0.1) * 0.5,
      'crash_prob': 0.05,
      'aeb_imminent': False,
      'ldw_warning': i == 50,  # One LDW event
    }
    
    car_state = {
      'vEgo': 20.0,
      'steeringTorque': 0.0,
    }
    
    actuators = {
      'accel': model_outputs['accel_pred'] + np.random.randn() * 0.1,
      'torque': model_outputs['torque_pred'] + np.random.randn() * 0.05,
    }
    
    metrics.update(model_outputs, car_state, actuators)
  
  # Print summary
  summary = metrics.get_summary()
  print("\nMetrics Summary:")
  for key, value in sorted(summary.items()):
    if isinstance(value, float):
      print(f"  {key:30s}: {value:8.4f}")
    else:
      print(f"  {key:30s}: {value}")
  
  # Print alerts
  alerts = metrics.get_alerts()
  if alerts:
    print("\nAlerts:")
    for alert in alerts:
      print(f"  ⚠️  {alert}")
  
  print("\n" + "=" * 50)
  print(f"✅ Comfort Score: {summary['comfort_score']:.1f}/100")
  print(f"✅ Smoothness Score: {summary['smoothness_score']:.1f}/100")
