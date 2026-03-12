# E2E Phase 2: Direct Actuation Implementation

## Overview

This implementation upgrades openpilot from an **E2E Planning** system to a true **E2E Control** system (Pixels-to-Control). The neural network now outputs direct actuator commands (steering torque, gas, brake) instead of just trajectory plans that require classical PID controllers.

## Key Changes

### 1. Model Interface Updates (`selfdrive/modeld/`)

**New Model Outputs:**
- `steer_torque_pred`: Steering torque prediction [-1, 1]
- `steer_angle_pred`: Steering angle prediction (radians)
- `gas_pred`: Gas pedal position [0, 1]
- `brake_pred`: Brake pedal position [0, 1]
- `crash_prob`: Crash probability for AEB [0, 1]
- `ttc_pred`: Time to collision (seconds)

**Files Modified:**
- `constants.py`: Added `E2EActuator` class with output slices
- `parse_model_outputs.py`: Added `parse_e2e_actuator_outputs()` method
- `modeld.py`: Extract E2E predictions and pass to `modelV2.action`
- `cereal/log.capnp`: Extended `ModelDataV2.Action` struct with new fields

### 2. Direct E2E Controllers (`selfdrive/controls/lib/`)

**New Controllers:**
- `LongControlE2E`: Bypasses PID controller, applies model's gas/brake predictions directly
- `LatControlE2E`: Bypasses PID/torque controllers, applies model's steering predictions directly

**Features:**
- Low-pass filtering for smooth actuation (0.1s longitudinal, 0.05s lateral)
- Vehicle limit checking
- AEB override capability

### 3. E2E Automatic Emergency Braking (AEB)

**Implementation:**
- Model outputs `crash_prob` and `ttc_pred` for collision detection
- AEB triggers when:
  - `crash_prob > 0.7`, OR
  - `0 < ttc_pred < 1.5` seconds
- Applies maximum braking (-4.0 m/s²) when triggered
- Integrated into `LongControlE2E.update()`

**Files Modified:**
- `controlsd.py`: AEB logic in `state_control()`
- `longcontrol_e2e.py`: AEB override handling

### 4. Simulator Support (`tools/sim/`)

**Updates:**
- `simulator.py`: Added E2E actuator state fields (`user_torque`, `user_gas`, `user_brake`, `aeb_active`)
- `simulated_car.py`: Pass through E2E commands to physics engine
- Supports both classical and E2E control modes

### 5. Configuration

**Enable E2E Mode:**
```bash
# Set parameter to enable E2E direct actuation
op params set E2E_Enabled true
```

**Fallback:**
When `E2E_Enabled` is false, the system uses classical PID/MPC controllers for backward compatibility.

## Architecture

### Classical Control (E2E_Enabled = false)
```
Model → Trajectory Plan → PID Controllers → Actuators
         (curvature,      (calculate error,
          acceleration)    apply gains)
```

### E2E Direct Control (E2E_Enabled = true)
```
Model → Direct Actuator Commands → Low-pass Filter → Actuators
         (torque, gas, brake)      (smooth commands)
```

## API Changes

### Model Output Schema
```capnp
struct ModelDataV2.Action {
  desiredCurvature @0 :Float32;      # Legacy
  desiredAcceleration @1 :Float32;    # Legacy
  shouldStop @2 :Bool;                # Legacy
  
  # E2E Phase 2: Direct actuator predictions
  steerTorquePred @3 :Float32;    # [-1, 1]
  steerAnglePred @4 :Float32;     # radians
  gasPred @5 :Float32;            # [0, 1]
  brakePred @6 :Float32;          # [0, 1]
  crashProb @7 :Float32;          # [0, 1]
  ttcPred @8 :Float32;            # seconds
}
```

### ControlsState Extensions
```capnp
struct ControlsState {
  e2EEnabled @62 :Bool;
  aebActive @63 :Bool;
  
  lateralControlState :union {
    e2EState @64 :LateralE2EState;
    # ... other states
  }
}

struct LateralE2EState {
  active @0 :Bool;
  version @1 :Int32;
  steeringAngleDesiredDeg @2 :Float32;
  outputTorque @3 :Float32;
  saturated @4 :Bool;
}
```

## Testing

### Simulator Testing
```bash
# Run simulator with E2E mode enabled
export E2E_Enabled=1
python tools/sim/run_bridge.py
```

### Validation Tests
1. **Lateral Stability**: Verify model's torque predictions maintain lane position
2. **Longitudinal Control**: Test smooth acceleration/braking following
3. **AEB Activation**: Verify emergency braking triggers correctly
4. **Vehicle Agnostic**: Test across different vehicle masses/configurations

## Benefits

### 1. Eliminated Tuning Overhead
- No per-vehicle PID gain tuning required
- Model learns vehicle dynamics from data
- Single model works across different car types

### 2. Reduced Latency
- Direct mapping from vision to actuators
- No intermediate PID calculation delays
- Faster response to changing conditions

### 3. Improved Performance
- Model optimizes for comfort and safety directly
- No error accumulation from multiple control stages
- Better handling of edge cases via AEB

### 4. Vehicle Agnostic
- Vehicle parameters (mass, wheelbase, etc.) fed into model
- Model learns appropriate torque for each vehicle type
- Deprecates need for `VehicleModel` calculations

## Migration Path

### Phase 1 (Current)
- Model outputs both trajectory plans AND actuator predictions
- Classical controllers still active by default
- E2E mode available via parameter flag

### Phase 2 (Future)
- Deprecate trajectory plan outputs
- E2E mode becomes default
- Remove classical PID controllers

### Phase 3 (Final)
- Complete removal of per-vehicle tuning files
- Model fully conditioned on vehicle parameters
- True end-to-end pixels-to-control system

## Safety Considerations

1. **Fallback System**: Classical controllers remain available if E2E fails
2. **AEB Override**: Vision-based emergency braking independent of main control
3. **Limit Checking**: All actuator commands clipped to vehicle-safe limits
4. **Monitoring**: E2E state logged for analysis and debugging

## Future Work

1. **Vehicle Conditioning**: Feed `CP.mass`, `CP.wheelbase`, etc. into model inputs
2. **Multi-Hypothesis Control**: Select best trajectory from model's hypotheses
3. **Learning-Based AEB**: Train model on near-miss and collision data
4. **Sim2Real Transfer**: Improve simulator physics for better real-world performance

## References

- Original Roadmap: `docs/contributing/roadmap.md`
- Model Architecture: `selfdrive/modeld/models/`
- Control Theory: `selfdrive/controls/lib/`
- Simulator: `tools/sim/README.md`
