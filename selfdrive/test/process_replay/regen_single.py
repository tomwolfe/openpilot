#!/usr/bin/env python3
"""Direct regeneration script for single segments without threading issues."""
import sys
from pathlib import Path

# Add openpilot to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.process_replay.regen import regen_and_save

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regen_single.py <route/segment>")
        print("Example: python regen_single.py regen755D8CB1E1F/2025-04-08--23-13-43--0/0")
        sys.exit(1)
    
    segment = sys.argv[1]
    # Parse route and segment number
    parts = segment.split("/")
    route = f"{parts[0]}/{parts[1]}"
    seg_num = int(parts[2]) if len(parts) > 2 else 0
    
    print(f"Regenerating: {route} segment {seg_num}")
    
    with OpenpilotPrefix():
        try:
            result = regen_and_save(route, seg_num, upload=False, dummy_driver_cam=True)
            print(f"Success: {result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
