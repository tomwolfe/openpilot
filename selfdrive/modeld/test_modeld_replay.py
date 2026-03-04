#!/usr/bin/env python3
"""
Quick modeld replay test for Phase 1 validation.

This runs modeld replay on a single segment to verify tinygrad integration works.
"""

import os
import sys
from pathlib import Path

# Set up environment before any imports
if '/usr/bin' not in os.environ.get('PATH', ''):
  os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
if os.environ.get('DEV') is None:
  os.environ['DEV'] = 'CPU'
if os.environ.get('CPU_LLVM') is None:
  os.environ['CPU_LLVM'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process


def test_modeld_replay():
  """Test modeld replay on a single segment."""
  # Use a test segment
  route = "regenAA0FC4ED71E|2025-04-08--22-57-50--0"
  route_base = route.rsplit('--', 1)[0]
  seg_num = route.rsplit('--', 1)[1]

  print(f"Testing modeld replay on route: {route}")

  # Load logreader
  print("Loading logs...")
  lr_path = get_url(route_base, seg_num, "rlog.zst")
  lr = list(LogReader(lr_path))
  print(f"Loaded {len(lr)} messages")

  # Load frame readers
  print("Loading frame readers...")
  frs = {
    'roadCameraState': FrameReader(get_url(route_base, seg_num, "fcamera.hevc")),
    'wideRoadCameraState': FrameReader(get_url(route_base, seg_num, "ecamera.hevc")),
  }
  print("Frame readers loaded")

  # Get modeld config
  modeld_cfg = get_process_config("modeld")

  # Run replay
  print("Running modeld replay...")
  output_logs = replay_process(modeld_cfg, lr, frs, disable_progress=False)

  print(f"\n✅ Modeld replay successful!")
  print(f"Generated {len(output_logs)} output messages")

  # Check output types
  output_types = set(m.which() for m in output_logs)
  print(f"Output message types: {output_types}")

  # Check for expected outputs
  expected = {'modelV2', 'drivingModelData', 'cameraOdometry'}
  if expected.issubset(output_types):
    print("✅ All expected output types present")
  else:
    missing = expected - output_types
    print(f"❌ Missing output types: {missing}")
    return False

  # Check execution times
  modelv2_msgs = [m for m in output_logs if m.which() == 'modelV2']
  if modelv2_msgs:
    exec_times = [m.modelV2.modelExecutionTime for m in modelv2_msgs if m.modelV2.modelExecutionTime > 0]
    if exec_times:
      avg_time = sum(exec_times) / len(exec_times)
      max_time = max(exec_times)
      print(f"\nExecution times:")
      print(f"  Average: {avg_time*1000:.2f}ms")
      print(f"  Max: {max_time*1000:.2f}ms")
      print(f"  Target: 40ms (for 20Hz)")

      if avg_time <= 0.04:  # 40ms
        print("✅ Meets 20Hz target (CPU mode)")
      else:
        print("⚠️  Does not meet 20Hz target (expected on CPU, test on TICI)")

  return True


if __name__ == "__main__":
  try:
    success = test_modeld_replay()
    sys.exit(0 if success else 1)
  except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
