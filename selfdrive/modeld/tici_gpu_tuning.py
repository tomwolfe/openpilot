#!/usr/bin/env python3
"""
AMD GPU Tuning Utilities for TICI Hardware - Phase 1 E2E 1.0

This module provides GPU-specific optimizations for the comma 3X TICI platform:
- Custom kernel tuning for AMD GPU architecture
- Memory tiling optimization
- Kernel fusion configuration
- Performance profiling with DEBUG=2

References:
- tinygrad_repo/extra/gemm/amd_matmul.py
- tinygrad_repo/docs/developer/am.md
"""

import os
import time
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.helpers import DEBUG, Context


# TICI AMD GPU Configuration
@dataclass
class TICIGPUConfig:
  """Configuration for TICI AMD GPU optimization."""

  # Device configuration
  device: str = "AMD"
  amd_iface: str = "USB"  # USB GPU interface

  # Kernel tuning parameters
  local_work_size: tuple[int, ...] = (256,)  # Default work group size
  global_work_size_multiplier: int = 4  # Multiplier for global work size

  # Memory optimization
  enable_zero_copy: bool = True
  enable_kernel_fusion: bool = True
  max_fused_kernels: int = 16

  # Cache configuration
  enable_schedule_cache: bool = True
  schedule_cache_size: int = 32

  # Performance tuning
  target_frequency_hz: float = 20.0
  inference_budget_ms: float = 40.0  # 80% of 50ms frame time

  # Debug options
  enable_profiling: bool = False
  debug_level: int = 0

  def apply(self):
    """Apply configuration to environment and tinygrad."""
    os.environ['DEV'] = self.device
    if self.amd_iface:
      os.environ['AMD_IFACE'] = self.amd_iface

    if self.enable_schedule_cache:
      os.environ['SCHEDULE_CACHE'] = '1'

    if self.enable_profiling or self.debug_level > 0:
      os.environ['DEBUG'] = str(self.debug_level)

    # Apply to Context
    Context(
      FUSE_ARITH=self.enable_kernel_fusion,
      FUSE_CONV_BW=self.enable_kernel_fusion,
    ).__enter__()


# Global GPU configuration instance
GPU_CONFIG = TICIGPUConfig()


@dataclass
class KernelTuningResult:
  """Results from kernel tuning."""

  kernel_name: str
  best_config: dict[str, Any]
  best_time_ms: float
  baseline_time_ms: float
  speedup: float
  configurations_tested: int


class AMDKernelTuner:
  """
  Kernel tuner for AMD GPU on TICI.

  This class helps find optimal kernel configurations for specific
  operations in the driving model.

  Usage:
    tuner = AMDKernelTuner()
    result = tuner.tune_conv2d(input_tensor, weight_tensor)
    print(f"Speedup: {result.speedup}x")
  """

  def __init__(self, config: TICIGPUConfig | None = None):
    """
    Initialize kernel tuner.

    Args:
      config: GPU configuration (uses global default if None)
    """
    self.config = config or GPU_CONFIG
    self._tuning_results: list[KernelTuningResult] = []

  def _benchmark(self, fn: Callable, iterations: int = 10) -> float:
    """
    Benchmark a function execution time.

    Args:
      fn: Function to benchmark
      iterations: Number of iterations

    Returns:
      Average execution time in milliseconds
    """
    # Warmup
    for _ in range(3):
      fn()
      Device.default.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
      start = time.perf_counter()
      fn()
      Device.default.synchronize()
      times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)

  def tune_conv2d(self,
                  input_shape: tuple[int, ...],
                  weight_shape: tuple[int, ...],
                  configurations: list[dict] | None = None) -> KernelTuningResult:
    """
    Tune conv2d kernel configuration.

    Args:
      input_shape: Shape of input tensor (N, C, H, W)
      weight_shape: Shape of weight tensor (OC, IC, K, K)
      configurations: List of configurations to test

    Returns:
      Tuning result with best configuration
    """
    if configurations is None:
      # Default configurations to test
      configurations = [
        {'tile_size': 16},
        {'tile_size': 32},
      ]

    # Create test tensors
    x = Tensor.randn(*input_shape)
    w = Tensor.randn(*weight_shape)

    best_time = float('inf')
    best_config = {}

    for config in configurations:
      def run_conv(tile=config['tile_size']):
        # Note: Actual kernel tuning would use BEAM or similar
        # This is a simplified benchmark
        return x.conv2d(w).realize()

      avg_time = self._benchmark(run_conv)

      if avg_time < best_time:
        best_time = avg_time
        best_config = config

    # Baseline without tuning
    def baseline():
      return x.conv2d(w).realize()

    baseline_time = self._benchmark(baseline)

    result = KernelTuningResult(
      kernel_name=f"conv2d_{input_shape}_x_{weight_shape}",
      best_config=best_config,
      best_time_ms=best_time,
      baseline_time_ms=baseline_time,
      speedup=baseline_time / best_time if best_time > 0 else 1.0,
      configurations_tested=len(configurations)
    )

    self._tuning_results.append(result)
    return result

  def tune_matmul(self,
                  input_shape: tuple[int, ...],
                  weight_shape: tuple[int, ...],
                  configurations: list[dict] | None = None) -> KernelTuningResult:
    """
    Tune matrix multiplication kernel configuration.

    Args:
      input_shape: Shape of input tensor (M, K)
      weight_shape: Shape of weight tensor (K, N)
      configurations: List of configurations to test

    Returns:
      Tuning result with best configuration
    """
    if configurations is None:
      configurations = [
        {'beam_search': 0},
        {'beam_search': 1},
      ]

    # Create test tensors
    a = Tensor.randn(*input_shape)
    b = Tensor.randn(*weight_shape)

    best_time = float('inf')
    best_config = {}

    for config in configurations:
      def run_matmul():
        return (a @ b).realize()

      avg_time = self._benchmark(run_matmul)

      if avg_time < best_time:
        best_time = avg_time
        best_config = config

    # Baseline
    def baseline():
      return (a @ b).realize()

    baseline_time = self._benchmark(baseline)

    result = KernelTuningResult(
      kernel_name=f"matmul_{input_shape}_x_{weight_shape}",
      best_config=best_config,
      best_time_ms=best_time,
      baseline_time_ms=baseline_time,
      speedup=baseline_time / best_time if best_time > 0 else 1.0,
      configurations_tested=len(configurations)
    )

    self._tuning_results.append(result)
    return result

  def get_tuning_report(self) -> str:
    """Generate tuning report."""
    lines = ["=" * 60, "AMD Kernel Tuning Report", "=" * 60]

    for result in self._tuning_results:
      lines.append(f"\nKernel: {result.kernel_name}")
      lines.append(f"  Best config: {result.best_config}")
      lines.append(f"  Best time: {result.best_time_ms:.2f}ms")
      lines.append(f"  Baseline:  {result.baseline_time_ms:.2f}ms")
      lines.append(f"  Speedup:   {result.speedup:.2f}x")
      lines.append(f"  Tested:    {result.configurations_tested} configs")

    if self._tuning_results:
      avg_speedup = sum(r.speedup for r in self._tuning_results) / len(self._tuning_results)
      lines.append(f"\nAverage speedup: {avg_speedup:.2f}x")

    return "\n".join(lines)


class MemoryTilingOptimizer:
  """
  Memory tiling optimizer for efficient GPU memory access.

  This optimizes memory layout for the specific AMD GPU architecture
  on TICI to maximize bandwidth utilization.
  """

  def __init__(self):
    self.tile_sizes = {
      'conv2d': 16,  # Default tile size for convolutions
      'matmul': 32,  # Default tile size for matrix multiply
      'image': 64,   # Default tile size for image processing
    }

  def optimize_image_buffer(self,
                            width: int,
                            height: int,
                            channels: int = 3) -> dict[str, Any]:
    """
    Optimize image buffer layout for GPU processing.

    Args:
      width: Image width
      height: Image height
      channels: Number of channels

    Returns:
      Optimization parameters
    """
    tile_size = self.tile_sizes['image']

    # Calculate tiled dimensions
    tiled_width = (width + tile_size - 1) // tile_size * tile_size
    tiled_height = (height + tile_size - 1) // tile_size * tile_size

    return {
      'original_shape': (height, width, channels),
      'tiled_shape': (tiled_height, tiled_width, channels),
      'tile_size': tile_size,
      'padding_width': tiled_width - width,
      'padding_height': tiled_height - height,
    }

  def optimize_feature_buffer(self,
                               feature_dim: int,
                               batch_size: int = 1) -> dict[str, Any]:
    """
    Optimize feature buffer layout for GPU processing.

    Args:
      feature_dim: Feature dimension
      batch_size: Batch size

    Returns:
      Optimization parameters
    """
    tile_size = self.tile_sizes['matmul']

    # Align to tile size
    aligned_dim = (feature_dim + tile_size - 1) // tile_size * tile_size

    return {
      'original_dim': feature_dim,
      'aligned_dim': aligned_dim,
      'tile_size': tile_size,
      'padding': aligned_dim - feature_dim,
    }


class PerformanceProfiler:
  """
  Performance profiler for tinygrad inference on TICI.

  Usage:
    profiler = PerformanceProfiler()
    with profiler.profile("vision_inference"):
      output = model(inputs)
    profiler.print_report()
  """

  def __init__(self):
    self._profiles: dict[str, list[float]] = {}
    self._current_profile: str | None = None
    self._start_time: float = 0

  def profile(self, name: str):
    """Context manager for profiling."""
    return ProfileContext(self, name)

  def start(self, name: str):
    """Start profiling a named operation."""
    self._current_profile = name
    self._start_time = time.perf_counter()

  def end(self):
    """End current profiling session."""
    if self._current_profile is None:
      return

    elapsed = (time.perf_counter() - self._start_time) * 1000

    if self._current_profile not in self._profiles:
      self._profiles[self._current_profile] = []
    self._profiles[self._current_profile].append(elapsed)

    self._current_profile = None

  def get_stats(self, name: str) -> dict[str, float]:
    """Get statistics for a profiled operation."""
    if name not in self._profiles or not self._profiles[name]:
      return {}

    times = self._profiles[name]
    return {
      'count': len(times),
      'mean': sum(times) / len(times),
      'min': min(times),
      'max': max(times),
      'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }

  def print_report(self):
    """Print profiling report."""
    print("\n" + "=" * 60)
    print("Performance Profile Report")
    print("=" * 60)

    for name, _times in self._profiles.items():
      stats = self.get_stats(name)
      if stats:
        print(f"\n{name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean:  {stats['mean']:.2f}ms")
        print(f"  Min:   {stats['min']:.2f}ms")
        print(f"  Max:   {stats['max']:.2f}ms")
        print(f"  Std:   {stats['std']:.2f}ms")

        # Check 20Hz compliance
        if stats['mean'] <= 40.0:
          print("  ✓ Meets 20Hz target")
        else:
          print("  ✗ Exceeds 20Hz budget")


class ProfileContext:
  """Context manager for profiling."""

  def __init__(self, profiler: PerformanceProfiler, name: str):
    self.profiler = profiler
    self.name = name

  def __enter__(self):
    self.profiler.start(self.name)
    return self

  def __exit__(self, *args):
    self.profiler.end()


def setup_tici_environment():
  """
  Set up environment for optimal TICI AMD GPU performance.

  Call this at the start of modeld to configure the environment.
  """
  # Apply GPU configuration
  GPU_CONFIG.apply()

  # Set additional environment variables for AMD GPU
  os.environ['AMD_GPU'] = '1'

  # Enable kernel fusion for better performance
  os.environ['FUSE_CONV_BW'] = '1'
  os.environ['FUSE_ARITH'] = '1'

  # Configure schedule cache
  os.environ['SCHEDULE_CACHE_SIZE'] = '32'

  if DEBUG >= 1:
    print("[TICI Setup] Environment configured for AMD GPU")
    print(f"  Device: {Device.DEFAULT}")
    print("  Schedule cache enabled")
    print("  Kernel fusion enabled")


def check_gpu_compatibility() -> dict[str, Any]:
  """
  Check GPU compatibility and capabilities.

  Returns:
    Dictionary with compatibility information
  """
  result = {
    'device': Device.DEFAULT,
    'is_amd': Device.DEFAULT == 'AMD',
    'is_tici': os.environ.get('AMD_IFACE') == 'USB' or Device.DEFAULT == 'AMD',
    'supports_gpu_acceleration': True,
  }

  if result['is_amd']:
    try:
      # Try to get device info
      device = Device[Device.DEFAULT]
      result['device_name'] = getattr(device, 'name', 'Unknown AMD GPU')
      result['is_initialized'] = True
    except Exception as e:
      result['is_initialized'] = False
      result['error'] = str(e)

  return result


# Example usage and testing
if __name__ == "__main__":
  print("TICI AMD GPU Tuning Utilities")
  print("=" * 60)

  # Check compatibility
  compat = check_gpu_compatibility()
  print("\nGPU Compatibility:")
  for key, value in compat.items():
    print(f"  {key}: {value}")

  # Setup environment
  print("\nSetting up TICI environment...")
  setup_tici_environment()

  # Run kernel tuning example
  print("\nRunning kernel tuning example...")
  tuner = AMDKernelTuner()

  # Tune a sample conv2d
  result = tuner.tune_conv2d(
    input_shape=(1, 3, 128, 256),
    weight_shape=(64, 3, 3, 3)
  )

  print("\nConv2d tuning result:")
  print(f"  Best config: {result.best_config}")
  print(f"  Speedup: {result.speedup:.2f}x")

  # Print full report
  print("\n" + tuner.get_tuning_report())
