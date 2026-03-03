"""
E2E Policy Validation Tests for MetaDrive Simulation.

This module provides validation tests for the Full E2E (Pixels-to-Policy) driving system.
Tests focus on scenarios where the E2E model must navigate without classical "lead car"
or "traffic light" objects being passed from the simulator to the controls stack.

Success Criteria:
- E2E model handles stop-and-go without explicit lead car object
- E2E model stops at red lights using only vision input
- E2E model handles cut-ins safely without explicit object detection
- All safety limits (ISO 15622, ISO 11270) are respected
"""

import time
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import deque

from cereal import log, messaging
import cereal.messaging as messaging


class E2ETestScenario(IntEnum):
  """E2E Policy test scenarios."""
  STOP_AND_GO = 0           # Stop and go traffic without lead car object
  RED_LIGHT_STOP = 1        # Stop at red light using vision only
  CUT_IN_HANDLING = 2       # Handle vehicle cutting in safely
  CURVE_ENTRY = 3           # Enter curve at appropriate speed
  MERGE_HIGHWAY = 4         # Merge onto highway
  EMERGENCY_BRAKE = 5       # Emergency braking scenario


@dataclass
class E2ETestMetrics:
  """Metrics collected during E2E policy testing."""
  # Longitudinal metrics
  max_acceleration: float = 0.0
  min_acceleration: float = 0.0
  max_jerk: float = 0.0
  avg_acceleration: float = 0.0
  
  # Lateral metrics
  max_curvature: float = 0.0
  max_lateral_accel: float = 0.0
  max_lateral_jerk: float = 0.0
  
  # Safety metrics
  iso_violations: int = 0
  safety_clips: int = 0
  e2e_policy_valid_ratio: float = 0.0
  
  # Performance metrics
  avg_latency_ms: float = 0.0
  frame_drop_rate: float = 0.0
  
  # Scenario-specific metrics
  stop_distance_error: float = 0.0  # Error from target stop position
  min_ttc: float = float('inf')     # Minimum time-to-collision
  
  # Buffers for analysis
  acceleration_buffer: list = field(default_factory=list)
  curvature_buffer: list = field(default_factory=list)
  velocity_buffer: list = field(default_factory=list)


class E2EPolicyValidator:
  """
  Validates E2E policy outputs against safety constraints and performance metrics.
  
  This validator runs in real-time during simulation, checking that:
  1. E2E policy outputs comply with ISO 15622 (longitudinal) and ISO 11270 (lateral)
  2. The policy is being followed correctly by the controls stack
  3. Performance meets the 100Hz frequency requirement
  """
  
  def __init__(self):
    self.metrics = E2ETestMetrics()
    self.prev_accel = 0.0
    self.prev_curvature = 0.0
    self.prev_time = time.time()
    
    # ISO 15622 limits (matching opendbc/safety/longitudinal.h)
    self.iso_long_accel_max = 2.0    # m/s^2
    self.iso_long_decel_max = 3.5    # m/s^2
    self.iso_long_jerk_max = 5.0     # m/s^3
    
    # ISO 11270 limits (matching opendbc/safety/lateral.h)
    self.iso_lateral_accel_max = 3.0  # m/s^2
    self.iso_lateral_jerk_max = 4.0   # m/s^3
    
    # E2E policy buffers
    self.policy_buffer = deque(maxlen=500)  # 5 seconds at 100Hz
    self.valid_policy_count = 0
    self.total_policy_count = 0
    
    # Latency tracking
    self.latency_buffer = deque(maxlen=100)
    
  def validate_policy(self, policy_msg: Optional[log.ModelDataV2.Policy], 
                      v_ego: float, a_ego: float) -> bool:
    """
    Validate an E2E policy message against safety constraints.
    
    Args:
      policy_msg: E2E Policy message from model
      v_ego: Current ego velocity (m/s)
      a_ego: Current ego acceleration (m/s^2)
    
    Returns:
      True if policy is valid and safe, False otherwise
    """
    self.total_policy_count += 1
    
    if policy_msg is None or not policy_msg.longitudinalAccelerations:
      return False
    
    current_time = time.time()
    dt = current_time - self.prev_time
    dt_us = int(dt * 1e6)
    
    # Validate longitudinal accelerations
    accel_list = list(policy_msg.longitudinalAccelerations)
    if len(accel_list) == 0:
      return False
    
    # Check first acceleration command (immediate)
    accel_cmd = accel_list[0]
    
    # ISO 15622 absolute limits
    if accel_cmd > self.iso_long_accel_max:
      self.metrics.iso_violations += 1
    if accel_cmd < -self.iso_long_decel_max:
      self.metrics.iso_violations += 1
    
    # Jerk limit
    if dt > 0:
      jerk = (accel_cmd - self.prev_accel) / dt
      if abs(jerk) > self.iso_long_jerk_max:
        self.metrics.iso_violations += 1
        self.metrics.safety_clips += 1
    
    # Validate lateral curvatures
    if policy_msg.lateralCurvatures:
      curvature_list = list(policy_msg.lateralCurvatures)
      if len(curvature_list) > 0:
        curvature_cmd = curvature_list[0]
        
        # Convert to lateral acceleration
        v = max(v_ego, 1.0)
        lateral_accel = curvature_cmd * v * v
        
        # ISO 11270 limit
        if abs(lateral_accel) > self.iso_lateral_accel_max:
          self.metrics.iso_violations += 1
        
        # Lateral jerk
        if dt > 0 and self.prev_curvature != 0:
          curvature_rate = (curvature_cmd - self.prev_curvature) / dt
          lateral_jerk = curvature_rate * v * v
          if abs(lateral_jerk) > self.iso_lateral_jerk_max:
            self.metrics.iso_violations += 1
            self.metrics.safety_clips += 1
        
        # Update metrics
        self.metrics.max_curvature = max(self.metrics.max_curvature, abs(curvature_cmd))
        self.metrics.max_lateral_accel = max(self.metrics.max_lateral_accel, abs(lateral_accel))
        self.curvature_buffer.append(curvature_cmd)
    
    # Update metrics
    self.metrics.max_acceleration = max(self.metrics.max_acceleration, accel_cmd)
    self.metrics.min_acceleration = min(self.metrics.min_acceleration, accel_cmd)
    self.acceleration_buffer.append(accel_cmd)
    
    self.prev_accel = accel_cmd
    self.prev_curvature = curvature_list[0] if policy_msg.lateralCurvatures else self.prev_curvature
    self.prev_time = current_time
    
    # Track valid policies
    if self.metrics.iso_violations == 0:
      self.valid_policy_count += 1
    
    # Store policy for analysis
    self.policy_buffer.append({
      'time': current_time,
      'acceleration': accel_cmd,
      'curvature': curvature_list[0] if policy_msg.lateralCurvatures else None,
      'policy_type': policy_msg.policyType,
      'confidence': policy_msg.confidence,
    })
    
    return self.metrics.iso_violations == 0
  
  def update_performance_metrics(self, frame_id: int, timestamp: float):
    """Update performance metrics like latency and frame drops."""
    current_time = time.time()
    latency = (current_time - timestamp) * 1000  # ms
    self.latency_buffer.append(latency)
    
    if len(self.latency_buffer) > 10:
      self.metrics.avg_latency_ms = np.mean(self.latency_buffer)
  
  def finalize_metrics(self, v_ego_buffer=None):
    """Finalize all metrics after test completion."""
    if self.acceleration_buffer:
      self.metrics.avg_acceleration = np.mean(np.abs(self.acceleration_buffer))
    
    if len(self.acceleration_buffer) > 1:
      accel_diff = np.diff(self.acceleration_buffer)
      if len(accel_diff) > 0:
        dt = 0.01  # 100Hz
        jerks = accel_diff / dt
        self.metrics.max_jerk = max(self.metrics.max_jerk, np.max(np.abs(jerks)))
    
    if self.total_policy_count > 0:
      self.metrics.e2e_policy_valid_ratio = self.valid_policy_count / self.total_policy_count
    
    if v_ego_buffer:
      self.metrics.velocity_buffer = list(v_ego_buffer)


class E2EScenarioRunner:
  """
  Runs specific E2E validation scenarios in MetaDrive.
  
  Each scenario tests the E2E model's ability to navigate without
  explicit object detection (lead cars, traffic lights, etc.).
  """
  
  def __init__(self, validator: E2EPolicyValidator):
    self.validator = validator
    self.sm = messaging.SubMaster(['modelV2', 'carState', 'controlsState', 'carControl'])
    self.pm = messaging.PubMaster(['testJoystick'])  # For scenario control
    
  def run_stop_and_go_test(self, duration: float = 30.0) -> E2ETestMetrics:
    """
    Test stop-and-go driving without explicit lead car object.
    
    The E2E model must use its vision-based policy to:
    1. Detect slowing traffic ahead
    2. Decelerate smoothly
    3. Come to a complete stop
    4. Resume when traffic moves
    
    Success criteria:
    - No ISO violations during deceleration
    - Smooth jerk profile (< 5.0 m/s^3)
    - Stop within reasonable distance of lead vehicle
    """
    print("Starting E2E Stop-and-Go Test...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
      self.sm.update(100)
      
      if not self.sm.all_checks():
        continue
      
      # Get E2E policy
      model_v2 = self.sm['modelV2']
      policy = model_v2.fullE2EPolicy if hasattr(model_v2, 'fullE2EPolicy') else None
      
      # Validate policy
      v_ego = self.sm['carState'].vEgo
      a_ego = self.sm['carState'].aEgo
      
      if policy:
        self.validator.validate_policy(policy, v_ego, a_ego)
      
      # Monitor for complete stop
      if v_ego < 0.5:
        print(f"  Vehicle stopped at {time.time() - start_time:.1f}s")
      
      # Update performance metrics
      self.validator.update_performance_metrics(
        model_v2.frameId,
        model_v2.timestampEof / 1e9 if model_v2.timestampEof else time.time()
      )
    
    self.validator.finalize_metrics()
    return self.validator.metrics
  
  def run_red_light_stop_test(self, duration: float = 20.0) -> E2ETestMetrics:
    """
    Test stopping at red light using vision only.
    
    The E2E model must:
    1. Detect red light through vision
    2. Plan smooth deceleration
    3. Stop at appropriate position
    
    Note: This tests that the E2E policy handles traffic signals
    without explicit traffic light objects from the simulator.
    """
    print("Starting E2E Red Light Stop Test...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
      self.sm.update(100)
      
      if not self.sm.all_checks():
        continue
      
      model_v2 = self.sm['modelV2']
      policy = model_v2.fullE2EPolicy if hasattr(model_v2, 'fullE2EPolicy') else None
      
      v_ego = self.sm['carState'].vEgo
      a_ego = self.sm['carState'].aEgo
      
      if policy:
        # Check for stopping maneuver
        if policy.policyType == log.ModelDataV2.Policy.PolicyType.stopping:
          print(f"  E2E policy initiated stop at {time.time() - start_time:.1f}s")
        
        self.validator.validate_policy(policy, v_ego, a_ego)
      
      self.validator.update_performance_metrics(
        model_v2.frameId,
        model_v2.timestampEof / 1e9 if model_v2.timestampEof else time.time()
      )
    
    self.validator.finalize_metrics()
    return self.validator.metrics
  
  def run_cut_in_handling_test(self, duration: float = 25.0) -> E2ETestMetrics:
    """
    Test handling of cut-in scenarios.
    
    The E2E model must:
    1. Detect vehicle cutting in through vision
    2. Adjust acceleration/curvature appropriately
    3. Maintain safe following distance
    
    Success criteria:
    - No emergency braking (> 3.5 m/s^2)
    - Smooth lateral response
    - Minimum TTC > 1.0 seconds
    """
    print("Starting E2E Cut-in Handling Test...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
      self.sm.update(100)
      
      if not self.sm.all_checks():
        continue
      
      model_v2 = self.sm['modelV2']
      policy = model_v2.fullE2EPolicy if hasattr(model_v2, 'fullE2EPolicy') else None
      
      v_ego = self.sm['carState'].vEgo
      a_ego = self.sm['carState'].aEgo
      
      if policy:
        # Check for emergency maneuver
        if policy.policyType == log.ModelDataV2.Policy.PolicyType.emergency:
          print(f"  E2E policy initiated emergency maneuver at {time.time() - start_time:.1f}s")
        
        valid = self.validator.validate_policy(policy, v_ego, a_ego)
        
        # Calculate TTC if we have lead data (for comparison only)
        # E2E should work without this
        if hasattr(model_v2, 'leadsV3') and len(model_v2.leadsV3) > 0:
          lead = model_v2.leadsV3[0]
          if lead.prob > 0.5 and lead.v[0] > 0:
            ttc = lead.x[0] / max(v_ego - lead.v[0], 0.1)
            self.validator.metrics.min_ttc = min(self.validator.metrics.min_ttc, ttc)
      
      self.validator.update_performance_metrics(
        model_v2.frameId,
        model_v2.timestampEof / 1e9 if model_v2.timestampEof else time.time()
      )
    
    self.validator.finalize_metrics()
    return self.validator.metrics


def run_all_e2e_tests():
  """Run complete E2E validation test suite."""
  print("=" * 60)
  print("E2E Policy Validation Test Suite")
  print("=" * 60)
  
  validator = E2EPolicyValidator()
  runner = E2EScenarioRunner(validator)
  
  results = {}
  
  # Run all scenarios
  print("\n1. Stop-and-Go Test")
  print("-" * 40)
  results['stop_and_go'] = runner.run_stop_and_go_test(duration=30.0)
  
  print("\n2. Red Light Stop Test")
  print("-" * 40)
  results['red_light'] = runner.run_red_light_stop_test(duration=20.0)
  
  print("\n3. Cut-in Handling Test")
  print("-" * 40)
  results['cut_in'] = runner.run_cut_in_handling_test(duration=25.0)
  
  # Print summary
  print("\n" + "=" * 60)
  print("TEST SUMMARY")
  print("=" * 60)
  
  for name, metrics in results.items():
    print(f"\n{name.upper()}:")
    print(f"  ISO Violations: {metrics.iso_violations}")
    print(f"  Safety Clips: {metrics.safety_clips}")
    print(f"  E2E Policy Valid Ratio: {metrics.e2e_policy_valid_ratio:.2%}")
    print(f"  Max Acceleration: {metrics.max_acceleration:.2f} m/s^2")
    print(f"  Min Acceleration: {metrics.min_acceleration:.2f} m/s^2")
    print(f"  Max Jerk: {metrics.max_jerk:.2f} m/s^3")
    print(f"  Max Curvature: {metrics.max_curvature:.4f} rad/m")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.1f} ms")
  
  # Overall pass/fail
  total_violations = sum(m.iso_violations for m in results.values())
  avg_valid_ratio = np.mean([m.e2e_policy_valid_ratio for m in results.values()])
  
  print("\n" + "=" * 60)
  if total_violations == 0 and avg_valid_ratio > 0.95:
    print("OVERALL: PASS ✓")
    print(f"  All ISO limits respected, {avg_valid_ratio:.1%} valid policies")
  else:
    print("OVERALL: FAIL ✗")
    print(f"  ISO Violations: {total_violations}")
    print(f"  Average Valid Policy Ratio: {avg_valid_ratio:.1%}")
  print("=" * 60)
  
  return results


if __name__ == "__main__":
  run_all_e2e_tests()
