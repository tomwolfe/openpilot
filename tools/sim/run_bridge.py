#!/usr/bin/env python3
import argparse

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

def create_bridge(dual_camera, high_quality, headless=False, test_duration=None, test_run=False):
  queue: Any = Queue()

  simulator_bridge = MetaDriveBridge(dual_camera, high_quality, test_duration=test_duration, test_run=test_run, headless=headless)
  simulator_process = simulator_bridge.run(queue)

  return queue, simulator_process, simulator_bridge

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display required)')
  parser.add_argument('--test_duration', type=float, default=None, help='Test duration in seconds (for automated testing)')
  parser.add_argument('--test_run', action='store_true', help='Run in test mode with automatic termination')

  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()

  queue, simulator_process, simulator_bridge = create_bridge(
    args.dual_camera, args.high_quality,
    headless=args.headless,
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
