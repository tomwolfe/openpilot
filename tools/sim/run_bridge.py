#!/usr/bin/env python3
import argparse

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

def create_bridge(dual_camera, high_quality, photorealistic=False, weather=None,
                  traffic_density=0.0, randomize_map=False, test_duration=None, test_run=False):
  queue: Any = Queue()

  simulator_bridge = MetaDriveBridge(
    dual_camera, high_quality,
    test_duration=test_duration if test_duration else float('inf'),
    test_run=test_run,
    photorealistic=photorealistic,
    weather=weather,
    traffic_density=traffic_density,
    randomize_map=randomize_map
  )
  simulator_process = simulator_bridge.run(queue)

  return queue, simulator_process, simulator_bridge

def main():
  args = parse_args()
  _, simulator_process, _ = create_bridge(
    args.dual_camera, args.high_quality,
    photorealistic=args.photorealistic,
    weather=args.weather,
    traffic_density=args.traffic_density,
    randomize_map=args.randomize_map
  )
  simulator_process.join()

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')

  # Enhanced simulation options
  parser.add_argument('--photorealistic', action='store_true',
                      help='Enable photorealistic rendering (requires GPU)')
  parser.add_argument('--weather', type=str, choices=['rain', 'fog', 'sunset', 'night'],
                      help='Weather/lighting conditions')
  parser.add_argument('--traffic_density', type=float, default=0.0,
                      help='Traffic density (0.0-1.0)')
  parser.add_argument('--randomize_map', action='store_true',
                      help='Randomize map layout for varied scenarios')
  parser.add_argument('--test_duration', type=int, default=None,
                      help='Test duration in seconds (for automated testing)')
  parser.add_argument('--test_run', action='store_true',
                      help='Run in test mode (auto-terminate on conditions)')

  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()

  queue, simulator_process, simulator_bridge = create_bridge(
    args.dual_camera, args.high_quality,
    photorealistic=args.photorealistic,
    weather=args.weather,
    traffic_density=args.traffic_density,
    randomize_map=args.randomize_map,
    test_duration=args.test_duration,
    test_run=args.test_run
  )

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(queue)
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(queue)

  simulator_bridge.shutdown()

  simulator_process.join()
