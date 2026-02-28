#!/usr/bin/env python3
"""
Deterministic Scenario Runner for openpilot SIL Testing.

This script runs automated Software-in-the-Loop (SIL) tests by:
1. Initializing a specific MetaDrive world (circular track or straight highway)
2. Automatically engaging openpilot via cruise control messaging
3. Running the simulation for a fixed duration
4. Monitoring onroadEvents and selfdriveState via SubMaster
5. Returning non-zero exit code on failures (collision, disable, off-roading)

Usage:
  PYTHONPATH=. python3 tools/sim/run_scenario.py --headless
  PYTHONPATH=. python3 tools/sim/run_scenario.py --scenario circular --duration 60
"""
import argparse
import os
import sys
import time
import signal
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.bridge.common import QueueMessageType


class ScenarioType(Enum):
  CIRCULAR = "circular"
  HIGHWAY = "highway"
  STRAIGHT = "straight"


@dataclass
class TestResult:
  """Result of a SIL test run."""
  success: bool = True
  duration: float = 0.0
  collision: bool = False
  soft_disable: bool = False
  immediate_disable: bool = False
  off_roading: bool = False
  vehicle_not_moving: bool = False
  timeout: bool = False
  errors: list[str] = field(default_factory=list)
  
  @property
  def exit_code(self) -> int:
    """Return non-zero exit code if any failure occurred."""
    if self.success:
      return 0
    return 1
  
  @property
  def failure_reasons(self) -> list[str]:
    """List all failure reasons."""
    reasons = []
    if self.collision:
      reasons.append("COLLISION detected")
    if self.soft_disable:
      reasons.append("SOFT_DISABLE event")
    if self.immediate_disable:
      reasons.append("IMMEDIATE_DISABLE event")
    if self.off_roading:
      reasons.append("OFF_ROADING detected")
    if self.vehicle_not_moving:
      reasons.append("VEHICLE_NOT_MOVING")
    if self.timeout:
      reasons.append("TIMEOUT")
    reasons.extend(self.errors)
    return reasons


class SILValidator:
  """Validates SIL test runs by monitoring openpilot state."""
  
  # Events that indicate a soft disable
  SOFT_DISABLE_EVENTS = {
    'steerUnavailable',
    'steerLimit',
    'gasUnavailable', 
    'brakeUnavailable',
    'cruiseUnavailable',
    'wrongCarMode',
    'driverDistracted',
    'driverUnresponsive',
    'preDriverDistracted',
    'preDriverUnresponsive',
    'modeldLagging',
    'locationdLagging',
    'posenetLagging',
    'controlsdLagging',
    'commIssue',
    'wrongCarMode',
    'pcmCruiseDisengaged',
  }
  
  # Events that indicate an immediate disable
  IMMEDIATE_DISABLE_EVENTS = {
    'steerFaultTemporary',
    'steerFaultPermanent',
    'steerFaultCritical',
    'brakeFault',
    'gasFault',
    'cruiseFault',
    'faultUnknown',
    'belowSteerSpeed',
    'carUnrecognized',
    'radarFault',
    'stockAeb',
    'steerTimeLimit',
  }
  
  def __init__(self, sm: messaging.SubMaster):
    self.sm = sm
    self.frame_count = 0
    self.engaged_frames = 0
    self.last_model_frame_id = -1
    self.model_frames_seen = 0
    
  def check_events(self) -> TestResult:
    """Check for any failure events in the current state."""
    result = TestResult()
    
    # Check onroadEvents
    if 'onroadEvents' in self.sm:
      for event in self.sm['onroadEvents']:
        event_name = event.name
        
        # Check for soft disable events
        if event.softDisable or event_name in self.SOFT_DISABLE_EVENTS:
          result.soft_disable = True
          result.errors.append(f"Soft disable event: {event_name}")
        
        # Check for immediate disable events
        if event.immediateDisable or event_name in self.IMMEDIATE_DISABLE_EVENTS:
          result.immediate_disable = True
          result.errors.append(f"Immediate disable event: {event_name}")
        
        # Check for noEntry events (prevent engagement)
        if event.noEntry:
          result.errors.append(f"No entry event: {event_name}")
    
    return result
  
  def check_model_vision(self) -> bool:
    """Check if modeld is processing frames."""
    if 'modelV2' not in self.sm:
      return False
    
    current_frame_id = self.sm['modelV2'].frameId
    if current_frame_id > self.last_model_frame_id:
      self.model_frames_seen += 1
      self.last_model_frame_id = current_frame_id
      return True
    return False
  
  def update(self) -> TestResult:
    """Update validator state and check for failures."""
    self.sm.update(0)
    self.frame_count += 1
    
    result = self.check_events()
    
    # Check if engaged
    if 'selfdriveState' in self.sm and self.sm['selfdriveState'].active:
      self.engaged_frames += 1
    
    # Check model vision
    self.check_model_vision()
    
    return result


def create_circular_map(track_size=60):
  """Create a circular track map configuration."""
  def straight_block(length):
    return {
      "id": "S",
      "pre_block_socket_index": 0,
      "length": length
    }

  def curve_block(length, angle=45, direction=0):
    return {
      "id": "C",
      "pre_block_socket_index": 0,
      "length": length,
      "radius": length,
      "angle": angle,
      "dir": direction
    }
  
  from metadrive.component.map.pg_map import MapGenerateMethod
  
  curve_len = track_size * 2
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
    ]
  )


def create_highway_map():
  """Create a straight highway map configuration."""
  def straight_block(length):
    return {
      "id": "S",
      "pre_block_socket_index": 0,
      "length": length
    }
  
  from metadrive.component.map.pg_map import MapGenerateMethod
  
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=4,
    lane_width=3.5,
    config=[
      None,
      straight_block(500),  # Long straight highway
    ]
  )


def create_straight_map():
  """Create a simple straight track."""
  def straight_block(length):
    return {
      "id": "S",
      "pre_block_socket_index": 0,
      "length": length
    }
  
  from metadrive.component.map.pg_map import MapGenerateMethod
  
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(200),
    ]
  )


def get_map_config(scenario: ScenarioType) -> dict:
  """Get map configuration for the specified scenario."""
  if scenario == ScenarioType.CIRCULAR:
    return create_circular_map()
  elif scenario == ScenarioType.HIGHWAY:
    return create_highway_map()
  else:  # STRAIGHT
    return create_straight_map()


def run_scenario(
  scenario: ScenarioType = ScenarioType.CIRCULAR,
  duration: float = 60.0,
  headless: bool = True,
  dual_camera: bool = True,
  high_quality: bool = False,
  verbose: bool = True
) -> TestResult:
  """
  Run a SIL scenario test.
  
  Args:
    scenario: Type of scenario to run
    duration: Test duration in seconds
    headless: Run without display
    dual_camera: Use dual camera setup
    high_quality: Use high quality rendering
    verbose: Print progress information
    
  Returns:
    TestResult with success/failure status and details
  """
  result = TestResult()
  start_time = time.monotonic()
  
  # Set up environment for headless mode
  if headless:
    os.environ['DISPLAY'] = ''
    os.environ['METADEBUG_HEADLESS'] = '1'
  
  # Create SubMaster for monitoring
  sm = messaging.SubMaster([
    'selfdriveState',
    'onroadEvents', 
    'modelV2',
    'carState',
    'controlsState',
    'managerState'
  ])
  
  # Create validator
  validator = SILValidator(sm)
  
  # Create bridge with test configuration
  map_config = get_map_config(scenario)
  
  # Build config for MetaDrive
  from metadrive.component.sensors.base_camera import _cuda_enable
  
  config = dict(
    use_render=not headless,
    render_offscreen=headless,
    vehicle_config=dict(
      enable_reverse=False,
      render_vehicle=False,
      image_source="rgb_road",
    ),
    sensors={
      "rgb_road": ("RGBCameraRoad", 1928, 1208)
    },
    image_on_cuda=_cuda_enable,
    image_observation=True,
    interface_panel=[],
    out_of_route_done=False,
    on_continuous_line_done=False,
    crash_vehicle_done=True,  # Enable collision detection
    crash_object_done=True,
    arrive_dest_done=False,
    traffic_density=0.0,
    map_config=map_config,
    decision_repeat=1,
    physics_world_step_size=5/100,
    preload_models=False,
    show_logo=False,
    anisotropic_filtering=False
  )
  
  if dual_camera:
    config["sensors"]["rgb_wide"] = ("RGBCameraWide", 1928, 1208)
  
  if verbose:
    print(f"Starting SIL scenario: {scenario.value}")
    print(f"Duration: {duration}s, Headless: {headless}, Dual Camera: {dual_camera}")
    print("-" * 60)
  
  # Launch openpilot processes
  manager_proc = None
  try:
    # Start the manager
    manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
    manager_proc = subprocess.Popen(
      [sys.executable, manager_path],
      env={**os.environ, 'SIMULATOR': '1'},
      cwd=BASEDIR
    )
    
    # Wait for manager to initialize
    if verbose:
      print("Waiting for openpilot manager to initialize...")
    
    max_wait = 30
    wait_start = time.monotonic()
    while time.monotonic() - wait_start < max_wait:
      sm.update(0)
      if sm.updated['managerState']:
        break
      time.sleep(0.1)
    
    # Create and start the bridge
    bridge = MetaDriveBridge(
      dual_camera=dual_camera,
      high_quality=high_quality,
      test_duration=duration,
      test_run=True,
      headless=headless
    )
    
    from multiprocessing import Queue
    queue: Queue = Queue()
    bridge_process = bridge.run(queue)
    
    if verbose:
      print("Bridge started, waiting for engagement...")
    
    # Wait for bridge to start
    max_wait = 30
    wait_start = time.monotonic()
    while not bridge.started.value and time.monotonic() - wait_start < max_wait:
      time.sleep(0.1)
      sm.update(0)
    
    if not bridge.started.value:
      result.success = False
      result.errors.append("Bridge failed to start")
      return result
    
    # Main test loop
    test_start = time.monotonic()
    last_print_time = test_start
    
    while time.monotonic() - test_start < duration:
      elapsed = time.monotonic() - test_start
      
      # Update validator
      validation_result = validator.update()
      
      # Check for failures
      if validation_result.soft_disable:
        result.soft_disable = True
        result.errors.extend(validation_result.errors)
      
      if validation_result.immediate_disable:
        result.immediate_disable = True
        result.errors.extend(validation_result.errors)
      
      # Check for collision via queue messages
      while not queue.empty():
        msg = queue.get()
        if msg.type == QueueMessageType.TERMINATION_INFO:
          done_info = msg.info
          if done_info:
            if done_info.get("crash_vehicle") or done_info.get("crash_object"):
              result.collision = True
            if done_info.get("out_of_lane"):
              result.off_roading = True
            if done_info.get("timeout"):
              result.timeout = True
            if done_info.get("vehicle_not_moving"):
              result.vehicle_not_moving = True
      
      # Check selfdriveState for engagement status
      if sm.updated.get('selfdriveState', False):
        sd_state = sm['selfdriveState']
        
        # Check for disable after being engaged
        if validator.engaged_frames > 10 and not sd_state.active:
          # Check if this was due to an event
          if sm.updated.get('onroadEvents', False):
            for event in sm['onroadEvents']:
              if event.name not in ('steerUnavailable', 'belowSteerSpeed'):
                result.soft_disable = True
                result.errors.append(f"Disengage event: {event.name}")
      
      # Periodic status update
      if verbose and elapsed - (last_print_time - test_start) >= 10:
        status = "ENGAGED" if sm['selfdriveState'].active else "DISENGAGED"
        model_ok = validator.model_frames_seen > 0
        print(f"  [{elapsed:5.1f}s] Status: {status}, Model frames: {validator.model_frames_seen}, Engaged frames: {validator.engaged_frames}")
        last_print_time = time.monotonic()
      
      # Early exit on critical failures
      if result.collision:
        if verbose:
          print("COLLISION detected, ending test early")
        break
      
      if result.immediate_disable:
        if verbose:
          print("IMMEDIATE_DISABLE detected, ending test early")
        break
      
      time.sleep(0.01)  # Small delay to prevent busy-waiting
    
    result.duration = time.monotonic() - start_time
    
    # Final validation
    if validator.engaged_frames < 10:
      result.success = False
      result.errors.append("Vehicle was not engaged for sufficient time")
    
    if validator.model_frames_seen < 10:
      result.success = False
      result.errors.append("Model vision not receiving frames")
    
    # Determine overall success
    if result.collision or result.soft_disable or result.immediate_disable or result.off_roading:
      result.success = False
    
    if verbose:
      print("-" * 60)
      if result.success:
        print(f"TEST PASSED: {scenario.value} scenario completed successfully")
      else:
        print(f"TEST FAILED: {scenario.value} scenario failed")
        for reason in result.failure_reasons:
          print(f"  - {reason}")
      print(f"Total duration: {result.duration:.2f}s")
      print(f"Engaged frames: {validator.engaged_frames}, Model frames: {validator.model_frames_seen}")
    
    # Cleanup
    bridge.shutdown()
    bridge_process.join(timeout=5)
    
  finally:
    # Cleanup manager
    if manager_proc is not None:
      manager_proc.terminate()
      manager_proc.wait(timeout=5)
  
  return result


def main():
  parser = argparse.ArgumentParser(description='SIL Scenario Runner for openpilot')
  parser.add_argument('--scenario', type=str, default='circular',
                      choices=['circular', 'highway', 'straight'],
                      help='Scenario type to run')
  parser.add_argument('--duration', type=float, default=60.0,
                      help='Test duration in seconds')
  parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (no display required)')
  parser.add_argument('--dual-camera', action='store_true', default=True,
                      help='Use dual camera setup')
  parser.add_argument('--no-dual-camera', action='store_true',
                      help='Disable dual camera setup')
  parser.add_argument('--high-quality', action='store_true',
                      help='Use high quality rendering')
  parser.add_argument('--quiet', action='store_true',
                      help='Suppress output')
  
  args = parser.parse_args()
  
  scenario = ScenarioType(args.scenario)
  
  result = run_scenario(
    scenario=scenario,
    duration=args.duration,
    headless=args.headless,
    dual_camera=not args.no_dual_camera,
    high_quality=args.high_quality,
    verbose=not args.quiet
  )
  
  sys.exit(result.exit_code)


if __name__ == "__main__":
  main()
