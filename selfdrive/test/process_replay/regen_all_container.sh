#!/usr/bin/env bash
set -e

# regen_all_container.sh - Run regen_all.py using Apple's container command instead of Docker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPILOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Container image to use
CONTAINER_IMAGE="${CONTAINER_IMAGE:-openpilot-local:latest}"

# Parse arguments
JOBS="${JOBS:-1}"
UPLOAD_FLAG=""
if [[ "${NO_UPLOAD:-}" != "1" ]]; then
  UPLOAD_FLAG="--upload"
fi

# Cars to process (can be overridden via environment)
WHITELIST_CARS="${WHITELIST_CARS:-}"
BLACKLIST_CARS="${BLACKLIST_CARS:-}"

echo "=========================================="
echo "Regenerating replay logs using container"
echo "=========================================="
echo "Container image: $CONTAINER_IMAGE"
echo "Jobs: $JOBS"
echo "Upload: ${UPLOAD_FLAG:-no}"
echo "Openpilot dir: $OPENPILOT_DIR"
echo ""

# Build container run command
# -v: Mount the openpilot directory
# -e: Pass through necessary environment variables
# --rm: Remove container after it exits
# -w: Set working directory

CONTAINER_CMD=(
  container run
  --rm
  -v "$OPENPILOT_DIR:$OPENPILOT_DIR"
  -w "$OPENPILOT_DIR"
  -e "OPENPILOT_DIR=$OPENPILOT_DIR"
  -e "PYTHONPATH=$OPENPILOT_DIR"
  -e "PYTHONUNBUFFERED=1"
  -e "TQDM_POSITION=-1"
  --memory=8g
  --cpus=4
)

# Add car filters if specified
if [[ -n "$WHITELIST_CARS" ]]; then
  CONTAINER_CMD+=(-e "WHITELIST_CARS=$WHITELIST_CARS")
fi
if [[ -n "$BLACKLIST_CARS" ]]; then
  CONTAINER_CMD+=(-e "BLACKLIST_CARS=$BLACKLIST_CARS")
fi

# Add the image
CONTAINER_CMD+=("$CONTAINER_IMAGE")

# Clean macOS-specific build artifacts before building for Linux
echo "Cleaning macOS build artifacts..."
find "$OPENPILOT_DIR" -name "*.so" -type f ! -path "*/.venv/*" -exec rm -f {} \; 2>/dev/null || true
# Also clean acados libs that are platform-specific
rm -rf "$OPENPILOT_DIR/third_party/acados/x86_64/lib/"* 2>/dev/null || true
rm -rf "$OPENPILOT_DIR/third_party/acados/larch64/lib/"* 2>/dev/null || true

# Copy Linux acados libraries from the container image
echo "Copying Linux acados libraries from container image..."
container run --rm --entrypoint "" "$CONTAINER_IMAGE" tar -C /home/batman/openpilot/third_party/acados -cf - x86_64/lib larch64/lib | tar -C "$OPENPILOT_DIR/third_party/acados" -xf -

# First, build native extensions inside the container
echo "Building native extensions inside container..."
"${CONTAINER_CMD[@]}" bash -c "scons -j\$(nproc)"

# Add the Python command
PYTHON_CMD=(
  python3
  selfdrive/test/process_replay/regen_all.py
  -j "$JOBS"
)

if [[ -z "$UPLOAD_FLAG" ]]; then
  PYTHON_CMD+=("--no-upload")
fi

# Add car filter arguments
if [[ -n "$WHITELIST_CARS" ]]; then
  PYTHON_CMD+=(--whitelist-cars $WHITELIST_CARS)
fi
if [[ -n "$BLACKLIST_CARS" ]]; then
  PYTHON_CMD+=(--blacklist-cars $BLACKLIST_CARS)
fi

# Run the container
echo "Running regen with container..."
echo ""

# Create a wrapper script that handles the signal issue for single-threaded execution
cat > /tmp/regen_wrapper.py << 'WRAPPER_EOF'
#!/usr/bin/env python3
import sys
import os
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobs", type=int, default=1)
parser.add_argument("--no-upload", action="store_true")
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
    from openpilot.selfdrive.test.process_replay.regen import regen_and_save
    from openpilot.selfdrive.test.process_replay.test_processes import FAKEDATA, source_segments as segments, EXCLUDED_PROCS
    from openpilot.tools.lib.route import SegmentName
    
    all_cars = {car for car, _ in segments}
    tested_cars = set(args.whitelist_cars) if args.whitelist_cars else all_cars
    tested_cars = tested_cars - set(args.blacklist_cars)
    tested_cars = {c.upper() for c in tested_cars}
    tested_segments = [(car, segment) for car, segment in segments if car in tested_cars]
    
    for segment in tested_segments:
        upload = not args.no_upload
        sn = SegmentName(segment[1])
        fake_dongle_id = 'regen' + ''.join(random.choice('0123456789ABCDEF') for _ in range(11))
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
WRAPPER_EOF

# Copy wrapper to openpilot directory so it can be accessed in the container
cp /tmp/regen_wrapper.py "$OPENPILOT_DIR/selfdrive/test/process_replay/regen_wrapper.py"

# Run the wrapper script instead
"${CONTAINER_CMD[@]}" python3 selfdrive/test/process_replay/regen_wrapper.py -j "$JOBS" ${UPLOAD_FLAG:+--no-upload} ${WHITELIST_CARS:+--whitelist-cars $WHITELIST_CARS} ${BLACKLIST_CARS:+--blacklist-cars $BLACKLIST_CARS}

# Clean up
rm -f "$OPENPILOT_DIR/selfdrive/test/process_replay/regen_wrapper.py"
rm -f /tmp/regen_wrapper.py

echo ""
echo "Regeneration complete!"
