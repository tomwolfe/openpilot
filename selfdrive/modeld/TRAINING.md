# E2E Model Training Pipeline

## Overview

This document describes the training pipeline for the End-to-End (E2E) driving model. The E2E model directly outputs actuator commands (steering torque, gas, brake) from camera inputs, eliminating the need for classical controllers.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      E2E Training Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Driving Logs → Preprocessing → Model Training → Validation      │
│     (rlog)        (normalize)    (tinygrad)     (metrics)        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Data Requirements

### Input Data Sources

1. **Camera Images** (from `roadCameraState`, `wideRoadCameraState`)
   - Resolution: 1928x1208 (road), 1928x1208 (wide)
   - Format: YUV420 semi-planar
   - Frequency: 20 Hz

2. **Vehicle State** (from `carState`)
   - Velocity (`vEgo`)
   - Steering angle (`steeringAngleDeg`)
   - Steering torque (`steeringTorque`)
   - Gas/brake pedal positions

3. **Navigation Data** (from `navEmbeddings`) - Optional
   - 64-dimensional embedding
   - Route information, maneuver instructions

4. **Vehicle Parameters** (from `CarParams`)
   - Mass, wheelbase, steerRatio, centerToFront
   - 4-dimensional embedding

### Target Labels

The model is trained to predict:

| Output | Range | Description |
|--------|-------|-------------|
| `steer_torque_pred` | [-1, 1] | Steering torque command |
| `steer_angle_pred` | radians | Steering angle command |
| `gas_pred` | [0, 1] | Gas pedal position |
| `brake_pred` | [0, 1] | Brake pedal position |
| `crash_prob` | [0, 1] | Crash probability |
| `ttc_pred` | seconds | Time to collision |
| `ldw_warning` | bool | Lane departure warning |

## Loss Functions

The E2E model uses multiple loss functions to ensure smooth, safe, human-like driving:

### 1. Actuation Loss (Primary)
```python
L_actuation = MSE(pred_torque, target_torque) + 
              MSE(pred_gas, target_gas) + 
              MSE(pred_brake, target_brake)
```
**Purpose**: Imitation learning - match human driver behavior

**Weight**: 2.0 (highest priority)

### 2. Jerk Loss
```python
L_jerk = MSE(d²/dt² pred_accel, 0)
```
**Purpose**: Passenger comfort - penalize rapid acceleration changes

**Weight**: 1.0

### 3. Comfort Loss
```python
L_comfort = max(0, |accel| - 2.0)² + max(0, |lat_accel| - 2.5)²
```
**Purpose**: Stay within comfortable acceleration bounds

**Weight**: 0.5

### 4. Steering Smoothness Loss
```python
L_smooth = MSE(d/dt pred_torque, 0)
```
**Purpose**: Eliminate high-frequency steering oscillations

**Weight**: 0.3

### 5. AEB Loss
```python
L_aeb = MSE(pred_brake, 0 | !emergency) + 
        2.0 * MSE(pred_brake, target_brake | emergency)
```
**Purpose**: Learn proper emergency braking behavior

**Weight**: 5.0 (safety-critical)

### Total Loss
```python
L_total = 2.0*L_actuation + 1.0*L_jerk + 0.5*L_comfort + 
          0.3*L_smooth + 5.0*L_aeb
```

## Training Procedure

### Step 1: Data Collection

Collect driving logs from human drivers:
```bash
# Record driving data
cd tools/replay
./replay --route "your_route_name"

# Data stored in:
# /data/media/0/realdata/<route_name>/<segment_number>/
```

### Step 2: Data Preprocessing

Extract and normalize training data:
```python
from openpilot.selfdrive.modeld.e2e_losses import create_training_example

# Load driving log
log_data = load_rlog("route/segment/rlog.bz2")

# Extract camera frames, vehicle state, actuator commands
# Normalize to [0, 1] or [-1, 1] range
# Create sequences of length T=10 (0.5 seconds at 20Hz)
```

### Step 3: Model Training

Train using tinygrad:
```python
from tinygrad.tensor import Tensor
from openpilot.selfdrive.modeld.e2e_losses import E2ELosses

# Initialize model
model = load_driving_model("driving_vision_tinygrad.pkl")

# Initialize losses
losses = E2ELosses()

# Training loop
for epoch in range(num_epochs):
  for batch in dataloader:
    # Forward pass
    outputs = model(batch['images'], batch['inputs'])
    
    # Compute loss
    total_loss, components = losses.compute_loss(
      model_outputs=outputs,
      human_targets=batch['targets'],
      car_state=batch['car_state'],
      return_components=True
    )
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Log metrics
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")
```

### Step 4: Validation

Validate on held-out data:
```python
# Compute validation metrics
val_metrics = {
  'mae_torque': mae(pred_torque, target_torque),
  'mae_gas': mae(pred_gas, target_gas),
  'mae_brake': mae(pred_brake, target_brake),
  'jerk_mean': mean(jerk(pred_accel)),
  'comfort_violations': pct(abs(accel) > 2.0),
}

# Compare against human driver baseline
assert val_metrics['mae_torque'] < human_std_torque * 1.2
```

### Step 5: Model Export

Export trained model:
```bash
# Convert to ONNX for deployment
python tools/model_export.py --input trained_model.pkl --output driving_policy.onnx

# Generate metadata
python selfdrive/modeld/get_model_metadata.py --model driving_policy.onnx
```

## Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Samples per batch |
| Sequence Length | 10 | Time steps (0.5s at 20Hz) |
| Learning Rate | 1e-4 | Adam optimizer LR |
| Weight Decay | 1e-5 | L2 regularization |
| Gradient Clip | 1.0 | Max gradient norm |
| Epochs | 50 | Training iterations |

## Quality Metrics

### Acceptance Criteria

Before deployment, the model must meet these criteria:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Torque MAE | < 0.05 | Mean absolute error vs human |
| Gas MAE | < 0.08 | Mean absolute error vs human |
| Brake MAE | < 0.05 | Mean absolute error vs human |
| Jerk (95th %ile) | < 2.0 m/s³ | Passenger comfort |
| Comfort Violations | < 1% | |accel| > 2.0 m/s² |
| AEB False Positives | < 0.1% | Unnecessary braking |
| AEB False Negatives | 0% | Missed emergencies |

### Simulator Validation

Test in MetaDrive simulator:
```bash
cd tools/sim
python run_bridge.py --model trained_model.pkl

# Test scenarios:
# 1. Lane keeping (100 km, no interventions)
# 2. Stop-and-go traffic (30 min, smooth following)
# 3. Highway merge (successful merges > 95%)
# 4. Emergency braking (AEB activation < 2s)
```

## Common Issues

### Issue 1: Jerky Steering

**Symptoms**: High-frequency steering oscillations

**Solution**: Increase `smoothness_weight` in loss function

```python
losses = E2ELosses(smoothness_weight=0.5)  # Increase from 0.3
```

### Issue 2: Aggressive Braking

**Symptoms**: Sudden, uncomfortable braking

**Solution**: Increase `comfort_weight` and add more braking data

```python
losses = E2ELosses(comfort_weight=1.0)  # Increase from 0.5
```

### Issue 3: Poor Navigation Following

**Symptoms**: Missing exits, wrong lane choices

**Solution**: Ensure nav_embeddings are properly trained

```python
# Check nav embedding integration
assert 'nav_embeddings' in model.input_names
# Retrain with navigation-conditioned data
```

## File Structure

```
selfdrive/modeld/
├── e2e_losses.py          # Loss function implementations
├── modeld.py              # Model inference
├── vehicle_conditioning.py # Vehicle parameter embedding
├── models/
│   ├── driving_vision_tinygrad.pkl  # Vision model weights
│   └── driving_policy_tinygrad.pkl  # Policy model weights
└── tests/
    └── test_e2e_losses.py  # Loss function tests

tools/
├── sim/                   # Simulator for validation
└── replay/                # Log replay for data extraction
```

## Next Steps

1. **Collect Training Data**: Gather diverse driving scenarios
2. **Train Initial Model**: Start with actuation loss only
3. **Add Regularization**: Incorporate jerk and comfort losses
4. **Validate in Simulator**: Test safety-critical scenarios
5. **Real-World Testing**: Closed-course validation
6. **Iterate**: Refine based on performance data

## References

- [E2E_SUMMARY.md](../../E2E_SUMMARY.md) - Implementation overview
- [e2e_losses.py](e2e_losses.py) - Loss function code
- [tinygrad_docs](../../tinygrad_repo/docs/) - Model training framework

---

**Last Updated**: 2026-03-12
**Status**: Ready for Training
**Version**: 1.0
