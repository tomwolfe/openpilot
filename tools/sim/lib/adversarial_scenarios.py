#!/usr/bin/env python3
"""
Adversarial Scenario Runner for openpilot E2E Policy Testing.

This script injects adversarial scenarios into the simulation to stress-test
the E2E policy. Scenarios include:
- Sudden cut-ins from adjacent lanes
- Disappearing lane markings
- Phantom obstacles
- Sensor noise/degradation

Usage:
  # Run with default scenarios
  python tools/sim/lib/adversarial_scenarios.py
  
  # Run specific scenario
  python tools/sim/lib/adversarial_scenarios.py --scenario cut_in
  
  # Run all scenarios with varying intensity
  python tools/sim/lib/adversarial_scenarios.py --all --intensity_range 0.2,0.8
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from openpilot.tools.sim.lib.world_model import WorldModel, AdversarialScenarioInjector, create_world_model


class ScenarioType(Enum):
  """Types of adversarial scenarios."""
  CUT_IN = "cut_in"
  DISAPPEARING_LANES = "disappearing_lanes"
  SENSOR_NOISE = "sensor_noise"
  PHANTOM_BRAKING = "phantom_braking"
  SUDDEN_STOP = "sudden_stop"


@dataclass
class ScenarioConfig:
  """Configuration for a scenario."""
  name: str
  type: ScenarioType
  intensity: float  # 0.0 to 1.0
  duration: float  # seconds
  delay: float  # seconds before triggering
  params: Dict[str, Any]


class AdversarialScenarioRunner:
  """
  Runs adversarial scenarios for E2E policy stress-testing.
  
  Usage:
    runner = AdversarialScenarioRunner()
    runner.add_scenario(ScenarioConfig(...))
    runner.run(world_model, simulator_state)
  """
  
  def __init__(self):
    self.scenarios: List[ScenarioConfig] = []
    self.active_scenario: Optional[ScenarioConfig] = None
    self.scenario_start_time: float = 0
    self.results: List[Dict[str, Any]] = []
    
    self.world_model: Optional[WorldModel] = None
    self.injector: Optional[AdversarialScenarioInjector] = None
  
  def add_scenario(self, config: ScenarioConfig):
    """Add a scenario to the queue."""
    self.scenarios.append(config)
  
  def initialize(self, initial_frame: np.ndarray):
    """Initialize world model with initial frame."""
    self.world_model = create_world_model()
    self.world_model.initialize(initial_frame)
    self.injector = AdversarialScenarioInjector(self.world_model)
  
  def run_scenario(self, config: ScenarioConfig, 
                   current_frame: np.ndarray) -> np.ndarray:
    """
    Run a single scenario.
    
    Args:
      config: Scenario configuration
      current_frame: Current camera frame
    
    Returns:
      Modified frame with scenario applied
    """
    params = config.params
    
    if config.type == ScenarioType.CUT_IN:
      return self.injector.inject_cut_in(
        current_frame,
        side=params.get('side', 'left'),
        intensity=config.intensity
      )
    
    elif config.type == ScenarioType.DISAPPEARING_LANES:
      return self.injector.inject_disappearing_lanes(
        current_frame,
        fade_factor=config.intensity
      )
    
    elif config.type == ScenarioType.SENSOR_NOISE:
      return self.injector.inject_sensor_noise(
        current_frame,
        noise_level=config.intensity
      )
    
    else:
      # Unknown scenario, return unchanged frame
      return current_frame
  
  def update(self, current_frame: np.ndarray, 
             elapsed_time: float) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Update scenario state and apply effects.
    
    Args:
      current_frame: Current camera frame
      elapsed_time: Time since scenario runner started
    
    Returns:
      Tuple of (modified_frame, status_dict)
    """
    status = {
      'active_scenario': None,
      'scenario_progress': 0.0,
      'scenario_complete': False,
    }
    
    # Check if we should trigger next scenario
    if self.active_scenario is None:
      for scenario in self.scenarios:
        if elapsed_time >= scenario.delay:
          self.active_scenario = scenario
          self.scenario_start_time = elapsed_time
          status['active_scenario'] = scenario.name
          break
    
    # Apply active scenario
    if self.active_scenario:
      elapsed_scenario = elapsed_time - self.scenario_start_time
      progress = elapsed_scenario / self.active_scenario.duration
      status['scenario_progress'] = min(progress, 1.0)
      
      if elapsed_scenario <= self.active_scenario.duration:
        # Scenario is active, apply it
        modified_frame = self.run_scenario(
          self.active_scenario,
          current_frame
        )
        
        # Update world model
        if self.world_model:
          self.world_model.update(modified_frame, np.zeros(3))
        
        status['active_scenario'] = self.active_scenario.name
      else:
        # Scenario complete
        status['scenario_complete'] = True
        self.results.append({
          'scenario': self.active_scenario.name,
          'duration': elapsed_scenario,
          'intensity': self.active_scenario.intensity,
        })
        self.active_scenario = None
        modified_frame = current_frame
    else:
      modified_frame = current_frame
    
    return modified_frame, status


def create_test_scenarios(intensity_range: tuple = (0.3, 0.8)) -> List[ScenarioConfig]:
  """Create a set of test scenarios."""
  min_intensity, max_intensity = intensity_range
  
  scenarios = [
    ScenarioConfig(
      name="cut_in_left_mild",
      type=ScenarioType.CUT_IN,
      intensity=min_intensity,
      duration=3.0,
      delay=2.0,
      params={'side': 'left'}
    ),
    ScenarioConfig(
      name="cut_in_right_strong",
      type=ScenarioType.CUT_IN,
      intensity=max_intensity,
      duration=3.0,
      delay=8.0,
      params={'side': 'right'}
    ),
    ScenarioConfig(
      name="disappearing_lanes_mild",
      type=ScenarioType.DISAPPEARING_LANES,
      intensity=min_intensity,
      duration=5.0,
      delay=15.0,
      params={}
    ),
    ScenarioConfig(
      name="disappearing_lanes_strong",
      type=ScenarioType.DISAPPEARING_LANES,
      intensity=max_intensity,
      duration=5.0,
      delay=22.0,
      params={}
    ),
    ScenarioConfig(
      name="sensor_noise_mild",
      type=ScenarioType.SENSOR_NOISE,
      intensity=min_intensity * 0.5,
      duration=4.0,
      delay=30.0,
      params={}
    ),
    ScenarioConfig(
      name="sensor_noise_strong",
      type=ScenarioType.SENSOR_NOISE,
      intensity=max_intensity * 0.5,
      duration=4.0,
      delay=36.0,
      params={}
    ),
  ]
  
  return scenarios


def main():
  parser = argparse.ArgumentParser(
    description='Adversarial Scenario Runner for E2E Policy Testing'
  )
  parser.add_argument(
    '--scenario',
    type=str,
    default='all',
    help='Scenario to run (cut_in, disappearing_lanes, sensor_noise, all)'
  )
  parser.add_argument(
    '--intensity',
    type=float,
    default=0.5,
    help='Scenario intensity (0.0 to 1.0)'
  )
  parser.add_argument(
    '--intensity_range',
    type=str,
    default='0.3,0.8',
    help='Intensity range for all scenarios (min,max)'
  )
  parser.add_argument(
    '--duration',
    type=float,
    default=60.0,
    help='Total test duration in seconds'
  )
  parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output file for results'
  )
  
  args = parser.parse_args()
  
  # Parse intensity range
  intensity_range = tuple(map(float, args.intensity_range.split(',')))
  
  print("=" * 80)
  print("Adversarial Scenario Runner - Phase 3 E2E Testing")
  print("=" * 80)
  print(f"Scenario: {args.scenario}")
  print(f"Intensity: {args.intensity}")
  print(f"Duration: {args.duration}s")
  print("=" * 80)
  
  # Create scenarios
  if args.scenario == 'all':
    scenarios = create_test_scenarios(intensity_range)
  else:
    scenario_type = ScenarioType(args.scenario)
    scenarios = [
      ScenarioConfig(
        name=f"{args.scenario}_test",
        type=scenario_type,
        intensity=args.intensity,
        duration=5.0,
        delay=2.0,
        params={}
      )
    ]
  
  print(f"Created {len(scenarios)} scenarios")
  
  # Initialize runner
  runner = AdversarialScenarioRunner()
  for scenario in scenarios:
    runner.add_scenario(scenario)
  
  # Create dummy frame for testing
  dummy_frame = np.random.randint(0, 255, (1208, 1928, 3), dtype=np.uint8)
  runner.initialize(dummy_frame)
  
  # Run simulation
  print("\nRunning scenarios...")
  start_time = time.time()
  
  frame_count = 0
  while time.time() - start_time < args.duration:
    elapsed = time.time() - start_time
    
    # Simulate frame update (in real use, this would come from simulator)
    current_frame = dummy_frame.copy()
    modified_frame, status = runner.update(current_frame, elapsed)
    
    if status['active_scenario']:
      print(f"  [{elapsed:.1f}s] Active: {status['active_scenario']} " +
            f"({status['scenario_progress']*100:.0f}%)")
    
    if status['scenario_complete']:
      print(f"  [{elapsed:.1f}s] ✓ Scenario complete")
    
    frame_count += 1
    time.sleep(0.05)  # Simulate 20Hz
  
  # Print results
  print("\n" + "=" * 80)
  print("Results:")
  print("=" * 80)
  
  for result in runner.results:
    print(f"  ✓ {result['scenario']}: {result['duration']:.2f}s, " +
          f"intensity={result['intensity']:.2f}")
  
  print(f"\nTotal frames processed: {frame_count}")
  print(f"Scenarios completed: {len(runner.results)}")
  
  # Save results if requested
  if args.output:
    import json
    with open(args.output, 'w') as f:
      json.dump({
        'scenarios': [r['scenario'] for r in runner.results],
        'total_frames': frame_count,
        'duration': args.duration,
      }, f, indent=2)
    print(f"Results saved to {args.output}")
  
  print("\n✓ Adversarial scenario test complete")


if __name__ == "__main__":
  main()
