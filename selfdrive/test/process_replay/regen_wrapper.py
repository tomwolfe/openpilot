#!/usr/bin/env python3
import sys
import os
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobs", type=int, default=1)
parser.add_argument("--upload", action="store_true")
parser.add_argument("--whitelist-cars", type=str, nargs="*", default=None)
parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[])
args = parser.parse_args()

# Set environment for the script
os.environ['JOBS'] = str(args.jobs)

# When running with JOBS=1, execute directly to avoid signal/threading issues
if args.jobs == 1:
    # Import and run regen_all directly without ThreadPoolExecutor
    import random
    import traceback
    from openpilot.common.prefix import OpenpilotPrefix
    from openpilot.common.params import Params
    from openpilot.selfdrive.test.process_replay.regen import regen_and_save
    from openpilot.selfdrive.test.process_replay.test_processes import FAKEDATA, segments, EXCLUDED_PROCS
    from openpilot.tools.lib.route import SegmentName

    all_cars = {car for car, _ in segments}
    tested_cars = set(args.whitelist_cars) if args.whitelist_cars else all_cars
    tested_cars = tested_cars - set(args.blacklist_cars)
    tested_cars = {c.upper() for c in tested_cars}
    tested_segments = [(car, segment) for car, segment in segments if car in tested_cars]

    for segment in tested_segments:
        # Clear cached parameters before each route to prevent car model mismatch
        # This removes state from previous routes that could cause conflicts
        params = Params()
        params.remove("CarParamsPrevRoute")
        params.remove("LiveParametersV2")
        params.remove("LiveDelay")
        
        upload = args.upload
        sn = SegmentName(segment[1])
        fake_dongle_id = ''.join(random.choice('0123456789ABCDEF') for _ in range(16))
        try:
            with OpenpilotPrefix():
                processes = [p for p in ["selfdrived", "controlsd", "card", "radard", "plannerd", "calibrationd",
                                         "dmonitoringd", "locationd", "paramsd", "lagd", "ubloxd", "torqued"]
                             if p not in EXCLUDED_PROCS]
                relr = regen_and_save(sn.route_name.canonical_name, sn.segment_num, processes=processes, upload=upload,
                                      outdir=os.path.join(FAKEDATA, fake_dongle_id), disable_tqdm=False, dummy_driver_cam=True)
                relr = '|'.join(relr.split('/')[-2:])
                print(f'  ("{segment[0]}", "{relr}"), ')
        except Exception as e:
            err = f"  {segment} failed: {str(e)}"
            err += traceback.format_exc()
            err += "\n\n"
            print(err)
else:
    # Run the original script with ThreadPoolExecutor
    exec(open('selfdrive/test/process_replay/regen_all.py').read())
