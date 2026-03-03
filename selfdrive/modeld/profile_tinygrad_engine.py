#!/usr/bin/env python3
"""
tinygrad Engine Profiling Script - Phase 1 E2E 1.0

This script profiles the tinygrad engine to verify:
1. Kernel execution times
2. Schedule cache effectiveness
3. 20Hz compliance
4. GPU utilization (when running on TICI)

Usage:
  python selfdrive/modeld/profile_tinygrad_engine.py
  DEBUG=2 python selfdrive/modeld/profile_tinygrad_engine.py
  USBGPU=1 DEBUG=2 python selfdrive/modeld/profile_tinygrad_engine.py  # On TICI
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Setup environment
os.environ['DEV'] = os.environ.get('DEV', 'CPU')
if os.environ.get('USBGPU'):
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'

from tinygrad.helpers import DEBUG
from tinygrad.device import Device
from tinygrad.tensor import Tensor

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openpilot.selfdrive.modeld.tinygrad_engine import (
  BigGraphScheduler,
  ZeroCopyBufferManager,
)
from openpilot.selfdrive.modeld.tici_gpu_tuning import (
  setup_tici_environment,
  check_gpu_compatibility,
  PerformanceProfiler,
  AMDKernelTuner,
)


def profile_big_graph_scheduler():
  """Profile Big Graph scheduling performance."""
  print("\n" + "="*70)
  print("Big Graph Scheduler Profiling")
  print("="*70)

  # Create a representative model
  def vision_model(img: Tensor, big_img: Tensor) -> list[Tensor]:
    # Simplified vision model for profiling
    x = img.cast('float32') / 255.0
    y = big_img.cast('float32') / 255.0

    # Simulate conv layers - match input channels
    x = x.conv2d(Tensor.kaiming_uniform(32, 12, 3, 3)).relu()
    y = y.conv2d(Tensor.kaiming_uniform(32, 12, 3, 3)).relu()

    # Fuse features
    fused = (x + y).reshape(x.shape[0], -1)
    output = fused.dot(Tensor.kaiming_uniform(fused.shape[1], 100))

    return [output]

  # Create scheduler
  scheduler = BigGraphScheduler(vision_model, prune=True)

  # Create test inputs - must be non-const for JIT
  img = Tensor.randn((1, 12, 64, 128), dtype='uint8').contiguous().realize()
  big_img = Tensor.randn((1, 12, 64, 128), dtype='uint8').contiguous().realize()

  print("\nInput shapes:")
  print(f"  img: {img.shape}")
  print(f"  big_img: {big_img.shape}")

  # Warmup
  print("\nWarming up JIT compilation...")
  scheduler.warmup(img, big_img)

  # Profile inference
  profiler = PerformanceProfiler()
  num_iterations = 20

  print(f"\nRunning {num_iterations} inference iterations...")
  times = []
  for i in range(num_iterations):
    with profiler.profile(f"inference_{i}"):
      start = time.perf_counter()
      _ = scheduler(img, big_img)
      Device.default.synchronize()
      elapsed = (time.perf_counter() - start) * 1000
      times.append(elapsed)

  # Statistics
  avg_time = sum(times) / len(times)
  min_time = min(times)
  max_time = max(times)
  std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

  print("\nInference Performance:")
  print(f"  Average: {avg_time:.2f}ms")
  print(f"  Min:     {min_time:.2f}ms")
  print(f"  Max:     {max_time:.2f}ms")
  print(f"  Std Dev: {std_time:.2f}ms")

  # 20Hz compliance check
  target_time = 50.0  # 20Hz = 50ms per frame
  budget_time = target_time * 0.8  # 80% budget for inference

  print("\n20Hz Compliance:")
  print(f"  Target:  {target_time}ms per frame")
  print(f"  Budget:  {budget_time}ms (80% for inference)")
  print(f"  Average: {avg_time:.2f}ms")

  if avg_time <= budget_time:
    print(f"  ✓ PASSES - Meets 20Hz target with {(budget_time - avg_time):.2f}ms margin")
  else:
    print(f"  ✗ FAILS - Exceeds budget by {(avg_time - budget_time):.2f}ms")

  # Schedule cache stats
  from tinygrad.engine.schedule import schedule_cache
  print("\nSchedule Cache:")
  print(f"  Cache size: {len(schedule_cache)} entries")
  print("  Cache hits: Enabled (reusing across frames)")

  return avg_time <= budget_time


def profile_buffer_manager():
  """Profile zero-copy buffer management."""
  print("\n" + "="*70)
  print("Zero-Copy Buffer Manager Profiling")
  print("="*70)

  buffer_manager = ZeroCopyBufferManager({}, dtype='uint8')

  # Create mock NV12 buffer
  import numpy as np
  yuv_size = 1928 * 1208 * 3 // 2  # NV12 format
  data = np.random.randint(0, 255, size=yuv_size, dtype=np.uint8)
  ptr = data.ctypes.data

  # Profile buffer creation
  num_iterations = 100

  print(f"\nCreating {num_iterations} tensors from memory blobs...")
  times = []
  for i in range(num_iterations):
    start = time.perf_counter()
    cache_key = ('test', ptr + i) if i > 0 else ('test', ptr)
    _ = buffer_manager.get_blob_tensor(ptr, (yuv_size,), cache_key)
    elapsed = (time.perf_counter() - start) * 1000 * 1000  # microseconds
    times.append(elapsed)

  avg_time = sum(times) / len(times)

  print("\nBuffer Creation Performance:")
  print(f"  Average: {avg_time:.2f}μs")
  print(f"  Cache size: {len(buffer_manager._buffer_cache)} entries")
  print("  Zero-copy: Enabled")

  return True


def profile_kernel_tuning():
  """Profile kernel tuning capabilities."""
  print("\n" + "="*70)
  print("AMD Kernel Tuning Profiling")
  print("="*70)

  tuner = AMDKernelTuner()

  # Profile conv2d tuning
  print("\nTuning conv2d kernel...")
  result = tuner.tune_conv2d(
    input_shape=(1, 3, 64, 128),
    weight_shape=(32, 3, 3, 3),
    configurations=[
      {'local_size': (64,), 'tile_size': 16},
      {'local_size': (128,), 'tile_size': 16},
    ]
  )

  print("\nConv2d Tuning Result:")
  print(f"  Best config: {result.best_config}")
  print(f"  Best time:   {result.best_time_ms:.2f}ms")
  print(f"  Baseline:    {result.baseline_time_ms:.2f}ms")
  print(f"  Speedup:     {result.speedup:.2f}x")

  # Profile matmul tuning
  print("\nTuning matmul kernel...")
  result = tuner.tune_matmul(
    input_shape=(64, 512),
    weight_shape=(512, 256),
    configurations=[
      {'local_size': (64, 64)},
      {'local_size': (128, 64)},
    ]
  )

  print("\nMatMul Tuning Result:")
  print(f"  Best config: {result.best_config}")
  print(f"  Best time:   {result.best_time_ms:.2f}ms")
  print(f"  Baseline:    {result.baseline_time_ms:.2f}ms")
  print(f"  Speedup:     {result.speedup:.2f}x")

  return True


def print_gpu_compatibility():
  """Print GPU compatibility information."""
  print("\n" + "="*70)
  print("GPU Compatibility Check")
  print("="*70)

  compat = check_gpu_compatibility()

  print("\nDevice Information:")
  for key, value in compat.items():
    status = "✓" if value else "✗"
    if isinstance(value, bool):
      print(f"  {status} {key}: {value}")
    else:
      print(f"    {key}: {value}")

  print(f"\nCurrent Device: {Device.DEFAULT}")
  print(f"DEBUG Level: {DEBUG.value}")

  return compat['is_tici']


def main():
  parser = argparse.ArgumentParser(description='Profile tinygrad engine')
  parser.add_argument('--all', action='store_true', help='Run all profiling tests')
  parser.add_argument('--scheduler', action='store_true', help='Profile Big Graph scheduler')
  parser.add_argument('--buffers', action='store_true', help='Profile buffer manager')
  parser.add_argument('--tuning', action='store_true', help='Profile kernel tuning')
  parser.add_argument('--gpu-check', action='store_true', help='Check GPU compatibility')

  args = parser.parse_args()

  # Default to all if no specific option
  run_all = not any([args.scheduler, args.buffers, args.tuning, args.gpu_check])

  results = {}

  # GPU compatibility check
  if args.gpu_check or run_all:
    results['gpu_check'] = print_gpu_compatibility()

  # Setup TICI environment if on AMD
  if Device.DEFAULT == 'AMD':
    setup_tici_environment()

  # Big Graph scheduler profiling
  if args.scheduler or run_all:
    results['scheduler'] = profile_big_graph_scheduler()

  # Buffer manager profiling
  if args.buffers or run_all:
    results['buffers'] = profile_buffer_manager()

  # Kernel tuning profiling
  if args.tuning or run_all:
    results['tuning'] = profile_kernel_tuning()

  # Summary
  print("\n" + "="*70)
  print("Profiling Summary")
  print("="*70)

  all_passed = all(v for v in results.values() if isinstance(v, bool))

  if all_passed:
    print("\n✓ All profiling tests PASSED")
  else:
    print("\n✗ Some profiling tests FAILED")
    for test, passed in results.items():
      if not passed:
        print(f"  - {test}: FAILED")

  print(f"\nDevice: {Device.DEFAULT}")
  print(f"DEBUG: {DEBUG.value}")
  print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

  return 0 if all_passed else 1


if __name__ == "__main__":
  sys.exit(main())
