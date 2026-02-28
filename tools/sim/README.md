openpilot in simulator
=====================

openpilot implements a [bridge](run_bridge.py) that allows it to run in the [MetaDrive simulator](https://github.com/metadriverse/metadrive).

## Launching openpilot
First, start openpilot.
``` bash
# Run locally
./tools/sim/launch_openpilot.sh
```

## Bridge usage
```
$ ./run_bridge.py -h
usage: run_bridge.py [-h] [--joystick] [--high_quality] [--dual_camera] [--headless] [--test_duration TEST_DURATION] [--test_run]
Bridge between the simulator and openpilot.

options:
  -h, --help            show this help message and exit
  --joystick
  --high_quality
  --dual_camera
  --headless            Run in headless mode (no display required)
  --test_duration TEST_DURATION
                        Test duration in seconds (for automated testing)
  --test_run            Run in test mode with automatic termination
```

#### Bridge Controls:
- To engage openpilot press 2, then press 1 to increase the speed and 2 to decrease.
- To disengage, press "S" (simulates a user brake)

#### All inputs:

```
| key  |   functionality       |
|------|-----------------------|
|  1   | Cruise Resume / Accel |
|  2   | Cruise Set    / Decel |
|  3   | Cruise Cancel         |
|  r   | Reset Simulation      |
|  i   | Toggle Ignition       |
|  q   | Exit all              |
| wasd | Control manually      |
```

## SIL (Software-in-the-Loop) Testing

### Automated Scenario Runner

The `run_scenario.py` script provides automated SIL testing with pass/fail validation:

```bash
# Run a 60-second circular track scenario in headless mode
PYTHONPATH=. python3 tools/sim/run_scenario.py --headless --scenario circular --duration 60

# Run a highway scenario
PYTHONPATH=. python3 tools/sim/run_scenario.py --headless --scenario highway --duration 120

# Run with dual camera disabled
PYTHONPATH=. python3 tools/sim/run_scenario.py --headless --no-dual-camera
```

#### Scenario Types:
- `circular` - Circular track with 4 turns (default)
- `highway` - Long straight highway with multiple lanes
- `straight` - Simple straight track

#### Exit Codes:
- `0` - Test passed (successful drive with no failures)
- `1` - Test failed due to:
  - **Collision** - Detected via MetaDrive's `crash_vehicle` or `crash_object` events
  - **Soft Disable** - Any event in `onroadEvents` with `softDisable` flag
  - **Immediate Disable** - Any event in `onroadEvents` with `immediateDisable` flag
  - **Off-Roading** - Vehicle leaves the drivable area (`out_of_lane`)
  - **Vehicle Not Moving** - Vehicle fails to move after engagement

#### Validation Suite:
The scenario runner monitors:
- `selfdriveState` - Engagement status
- `onroadEvents` - Disengage events
- `modelV2` - Vision processing (frameId must increment)
- MetaDrive simulation events (collision, off-road, timeout)

### Headless Mode

Both `run_bridge.py` and `run_scenario.py` support headless operation for CI/CD:

```bash
# Headless bridge with test duration
./run_bridge.py --headless --test_run --test_duration 60

# Headless scenario runner (recommended for automated testing)
PYTHONPATH=. python3 tools/sim/run_scenario.py --headless
```

Headless mode uses MetaDrive's off-screen rendering to generate valid YUV frames for `camerad` and sensor data for `sensord` without requiring a display.

### Mock Sensor Parity

The simulator provides realistic, noisy sensor data:
- **IMU**: Accelerometer and gyroscope with Gaussian noise and bias
- **GPS**: Lat/lon/altitude with realistic noise characteristics
- **Cameras**: YUV frames via `VisionIpcServer` for `modeld` processing

## MetaDrive

### Launching Metadrive
Start bridge processes located in tools/sim:
``` bash
./run_bridge.py
```

### Running in Docker

The SIL suite can run inside `Dockerfile.openpilot` without a physical GPU:

```bash
docker build -f Dockerfile.openpilot -t openpilot-sil .
docker run --rm openpilot-sil PYTHONPATH=. python3 tools/sim/run_scenario.py --headless --duration 60
```

For systems without a display, use `xvfb-run` as a fallback:
```bash
xvfb-run -a PYTHONPATH=. python3 tools/sim/run_scenario.py --duration 60
```