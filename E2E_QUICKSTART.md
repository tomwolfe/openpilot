# E2E Phase 2: Quick Start Guide

## Overview
This implementation transforms openpilot from an **E2E Planning** system to a true **E2E Control** system where the neural network outputs direct actuator commands (steering torque, gas, brake) instead of trajectory plans.

## What's Been Implemented

### ✅ Step 1: Model Interface Updates
- **New model outputs**: `steer_torque_pred`, `steer_angle_pred`, `gas_pred`, `brake_pred`, `crash_prob`, `ttc_pred`
- **Schema updates**: Extended `ModelDataV2.Action` in capnp
- **Parser updates**: Added `parse_e2e_actuator_outputs()` method

### ✅ Step 2: E2E Controllers
- **`LongControlE2E`**: Direct longitudinal control without PID
- **`LatControlE2E`**: Direct lateral control without PID/torque controllers
- **Low-pass filtering**: Smooth actuation (0.1s longitudinal, 0.05s lateral)

### ✅ Step 3: Vehicle-Agnostic Conditioning
- **`VehicleConditioning`**: Extracts vehicle parameters (mass, wheelbase, steer ratio)
- **Normalized embedding**: Fed into model for vehicle-specific learning
- **Auto-initialization**: Integrated into modeld pipeline

### ✅ Step 4: E2E AEB System
- **Vision-based AEB**: Uses model's `crash_prob` and `ttc_pred` outputs
- **Automatic override**: Applies -4.0 m/s² braking when collision imminent
- **Safety integration**: Works with panda safety system

### ✅ Step 5: Simulator Support
- **Direct actuation mode**: Simulator accepts torque/gas/brake commands
- **Backward compatible**: Works with both classical and E2E control
- **Physics integration**: Commands passed to MetaDrive physics engine

## Files Created/Modified

### New Files
```
selfdrive/controls/lib/longcontrol_e2e.py       # E2E longitudinal controller
selfdrive/controls/lib/latcontrol_e2e.py        # E2E lateral controller
selfdrive/modeld/vehicle_conditioning.py        # Vehicle parameter embedding
E2E_PHASE2_IMPLEMENTATION.md                    # Detailed documentation
```

### Modified Files
```
cereal/log.capnp                                # Added E2E schema fields
selfdrive/modeld/constants.py                   # Added E2EActuator slices
selfdrive/modeld/parse_model_outputs.py         # Added E2E parser
selfdrive/modeld/modeld.py                      # Integrated vehicle conditioning
selfdrive/controls/controlsd.py                 # E2E controller selection
tools/sim/car_interface/simulator.py            # E2E actuator support
tools/sim/lib/simulated_car.py                  # E2E command passthrough
```

## Enabling E2E Mode

### Method 1: Parameter Setting
```bash
# Enable E2E direct actuation
op params set E2E_Enabled true

# Verify setting
op params get E2E_Enabled
```

### Method 2: Code Override
In `controlsd.py`, temporarily set:
```python
self.e2e_enabled = True  # Force E2E mode
```

## Architecture Comparison

### Before (Classical Control)
```
Vision Model
    ↓
Trajectory Plan (curvature, acceleration)
    ↓
PID Controllers (calculate error, apply gains)
    ↓
Actuators (torque, gas, brake)
```

### After (E2E Direct Control)
```
Vision Model + Vehicle Parameters
    ↓
Direct Actuator Predictions (torque, gas, brake)
    ↓
Low-pass Filter (smooth commands)
    ↓
Actuators (direct application)
```

## Testing

### Simulator Testing
```bash
# 1. Enable E2E mode
export E2E_Enabled=1

# 2. Run simulator
cd tools/sim
python run_bridge.py

# 3. Monitor E2E state
cabana tools/zmqlogger --filter controlsState.e2EEnabled
```

### Real Vehicle Testing (CAUTION: Experimental!)
```bash
# WARNING: This is experimental software
# Test only in controlled, safe environments

# 1. Enable E2E mode
op params set E2E_Enabled true

# 2. Start openpilot
launch_openpilot.sh

# 3. Monitor logs for E2E activity
tail -f /data/log/*/modeld.log | grep -i e2e
```

## Monitoring & Debugging

### Check E2E Status
```python
import cereal.messaging as messaging
sm = messaging.SubMaster(['controlsState', 'modelV2'])

while True:
  sm.update(0)
  print(f"E2E Enabled: {sm['controlsState'].e2EEnabled}")
  print(f"AEB Active: {sm['controlsState'].aebActive}")
  print(f"Steer Torque Pred: {sm['modelV2'].action.steerTorquePred}")
  print(f"Gas Pred: {sm['modelV2'].action.gasPred}")
  print(f"Brake Pred: {sm['modelV2'].action.brakePred}")
```

### Log Analysis
```bash
# Search for E2E-related logs
grep -r "E2E" /data/log/*/modeld.log
grep -r "e2e_enabled" /data/log/*/controlsd.log

# Monitor AEB activations
grep -r "crash_prob\|ttc_pred" /data/log/*/modeld.log
```

## Expected Behavior

### Normal Driving
- Smooth steering following model's torque predictions
- Gradual acceleration/braking from model's pedal predictions
- No PID oscillations or tuning-related issues

### AEB Activation
When model detects imminent collision:
- `crash_prob > 0.7` OR `ttc_pred < 1.5s`
- Immediate maximum braking (-4.0 m/s²)
- AEB flag set in controlsState

### Fallback to Classical
If E2E mode disabled or model outputs invalid:
- System automatically uses classical PID controllers
- No manual intervention required

## Known Limitations

1. **Model Training Required**: Current models don't output actuator predictions
   - Need to train with new output heads
   - Use existing trajectory outputs as fallback

2. **Simulator Physics**: MetaDrive may not perfectly match real vehicle dynamics
   - Validate in controlled real-world testing
   - Tune filter time constants as needed

3. **Vehicle Conditioning**: Embedding integration requires model architecture changes
   - Current implementation provides framework
   - Model training needed to utilize conditioning

## Next Steps

### For Developers
1. **Train E2E Model**: Add actuator output heads to driving model
2. **Collect Data**: Gather torque/pedal labels from human drivers
3. **Validate**: Test in simulator across various scenarios
4. **Tune**: Adjust filter constants for optimal performance

### For Users
1. **Test in Simulator**: Try E2E mode in MetaDrive simulation
2. **Report Issues**: Log any unexpected behavior
3. **Contribute Data**: Help collect training data for E2E model

## Safety Warnings

⚠️ **EXPERIMENTAL SOFTWARE**
- This implementation is for research and development
- Do NOT use on public roads without extensive testing
- Always maintain readiness to take manual control
- Test only in controlled, safe environments

⚠️ **AEB Testing**
- AEB system may trigger false positives during development
- Verify braking behavior in safe locations first
- Monitor crash_prob and ttc_pred values

## Support & Documentation

- **Full Documentation**: `E2E_PHASE2_IMPLEMENTATION.md`
- **Code Comments**: Check individual controller files
- **Issues**: Report bugs via GitHub issues
- **Discussion**: openpilot Discord #e2e-control channel

## Credits

Implementation based on openpilot roadmap for E2E Phase 2:
- Direct actuation (pixels-to-control)
- Vehicle-agnostic learning
- Vision-based AEB
- Elimination of per-car tuning

---

**Version**: 1.0  
**Date**: 2026-03-12  
**Status**: Experimental - Development Build
