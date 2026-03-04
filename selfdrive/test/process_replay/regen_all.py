#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import random
import traceback
from tqdm import tqdm

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.process_replay.regen import regen_and_save
from openpilot.selfdrive.test.process_replay.test_processes import FAKEDATA, segments, EXCLUDED_PROCS
from openpilot.tools.lib.route import SegmentName
from openpilot.common.params import Params


def regen_job(segment, upload, disable_tqdm):
  # Clear cached parameters before each route to prevent car model mismatch
  # This removes state from previous routes that could cause conflicts
  params = Params()
  params.remove("CarParamsPrevRoute")
  params.remove("LiveParametersV2")
  params.remove("LiveDelay")
  
  with OpenpilotPrefix():
    sn = SegmentName(segment[1])
    fake_dongle_id = 'regen' + ''.join(random.choice('0123456789ABCDEF') for _ in range(11))
    try:
      # Exclude modeld and dmonitoringmodeld as they require specific model files
      processes = [p for p in ["selfdrived", "controlsd", "card", "radard", "plannerd", "calibrationd",
                               "dmonitoringd", "locationd", "paramsd", "lagd", "ubloxd", "torqued"]
                   if p not in EXCLUDED_PROCS]
      relr = regen_and_save(sn.route_name.canonical_name, sn.segment_num, processes=processes, upload=upload,
                            outdir=os.path.join(FAKEDATA, fake_dongle_id), disable_tqdm=disable_tqdm, dummy_driver_cam=True)
      relr = '|'.join(relr.split('/')[-2:])
      return f'  ("{segment[0]}", "{relr}"), '
    except Exception as e:
      err = f"  {segment} failed: {str(e)}"
      err += traceback.format_exc()
      err += "\n\n"
      return err


if __name__ == "__main__":
  all_cars = {car for car, _ in segments}

  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("-j", "--jobs", type=int, default=1)
  parser.add_argument("--no-upload", action="store_true")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=all_cars,
                      help="Whitelist given cars from the test (e.g. HONDA)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[],
                      help="Blacklist given cars from the test (e.g. HONDA)")
  args = parser.parse_args()

  tested_cars = set(args.whitelist_cars) - set(args.blacklist_cars)
  tested_cars = {c.upper() for c in tested_cars}
  tested_segments = [(car, segment) for car, segment in segments if car in tested_cars]

  # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid /dev/shm semaphore issues
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    p = pool.map(regen_job, tested_segments, [not args.no_upload] * len(tested_segments), [args.jobs > 1] * len(tested_segments))
    msg = "Copy these new segments into test_processes.py:"
    for seg in tqdm(p, desc="Generating segments", total=len(tested_segments)):
      msg += "\n" + str(seg)
    print()
    print()
    print(msg)
