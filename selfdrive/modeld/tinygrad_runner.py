#!/usr/bin/env python3
"""
Tinygrad Runner for Phase 1 E2E 1.0 - Unified Inference Backend

This module provides a clean, unified interface for tinygrad-based model inference,
integrating the existing tinygrad_engine.py components with modeld.py.

Key Features:
- Drop-in replacement for legacy inference backend
- Big Graph scheduling with TinyJit for 20Hz real-time performance
- Zero-copy buffer management for VisionIPC integration
- Hardware-aware backend selection (Qualcomm/AMD/CPU)
- Schedule cache reuse across frames
- Performance profiling and 20Hz compliance checking

Usage:
  from openpilot.selfdrive.modeld.tinygrad_runner import TinygradRunner

  runner = TinygradRunner(
    vision_pkl_path=...,
    policy_pkl_path=...,
    vision_metadata_path=...,
    policy_metadata_path=...,
    models_dir=...
  )

  # For each frame:
  vision_output = runner.run_vision(bufs, transforms)
  policy_output = runner.run_policy(policy_inputs)
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context, DEBUG, getenv

from openpilot.common.file_chunker import read_file_chunked
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.tici_gpu_tuning import (
  setup_tici_environment,
  check_gpu_compatibility,
  GPU_CONFIG,
  PerformanceProfiler,
)


@dataclass
class RunnerConfig:
  """Configuration for TinygradRunner."""

  # Model paths
  vision_pkl_path: Path | None = None
  policy_pkl_path: Path | None = None
  vision_metadata_path: Path | None = None
  policy_metadata_path: Path | None = None
  models_dir: Path | None = None

  # Performance settings
  enable_schedule_cache: bool = True
  enable_kernel_fusion: bool = True
  warmup_runs: int = 3
  target_frequency_hz: float = 20.0

  # Debug settings
  enable_profiling: bool = False
  debug_level: int = 0

  def __post_init__(self):
    """Set default paths if not specified."""
    if self.models_dir is None:
      self.models_dir = Path(__file__).parent / 'models'

    if self.vision_pkl_path is None:
      self.vision_pkl_path = self.models_dir / 'driving_vision_tinygrad.pkl'

    if self.policy_pkl_path is None:
      self.policy_pkl_path = self.models_dir / 'driving_policy_tinygrad.pkl'

    if self.vision_metadata_path is None:
      self.vision_metadata_path = self.models_dir / 'driving_vision_metadata.pkl'

    if self.policy_metadata_path is None:
      self.policy_metadata_path = self.models_dir / 'driving_policy_metadata.pkl'


class ModelLoadError(Exception):
  """Raised when model files cannot be loaded."""


class TinygradRunner:
  """
  Unified tinygrad inference runner for openpilot modeld.

  This class provides a clean interface for running vision and policy models
  using tinygrad with optimized scheduling and zero-copy buffer management.

  Architecture:
  - Vision Model: Processes camera images -> features, pose, lane lines, etc.
  - Policy Model: Processes features + desires -> trajectory plan, actions

  Performance:
  - Uses TinyJit for kernel fusion and reduced GPU-CPU overhead
  - Big Graph scheduling for entire model execution
  - Zero-copy transfers from VisionIPC buffers
  - Schedule cache reuse across frames
  - Target: 20Hz inference (50ms budget, 40ms for inference)
  """

  def __init__(self, config: RunnerConfig | None = None):
    """
    Initialize the tinygrad runner.

    Args:
      config: Runner configuration (uses defaults if None)
    """
    self.config = config or RunnerConfig()

    # Setup GPU environment (auto-detects hardware)
    setup_tici_environment()

    # Load metadata
    self._load_metadata()

    # Load models
    self._load_models()

    # Initialize state
    self._init_state()

    # Performance profiler
    self.profiler = PerformanceProfiler() if self.config.enable_profiling else None

    # Warmup
    if self.config.warmup_runs > 0:
      self._warmup()

  def _load_metadata(self):
    """Load model metadata."""
    with open(self.config.vision_metadata_path, 'rb') as f:
      vision_meta = pickle.load(f)
      self.vision_input_shapes = vision_meta['input_shapes']
      self.vision_input_names = list(self.vision_input_shapes.keys())
      self.vision_output_slices = vision_meta['output_slices']
      self.vision_output_size = vision_meta['output_shapes']['outputs'][1]

    with open(self.config.policy_metadata_path, 'rb') as f:
      policy_meta = pickle.load(f)
      self.policy_input_shapes = policy_meta['input_shapes']
      self.policy_output_slices = policy_meta['output_slices']
      self.policy_output_size = policy_meta['output_shapes']['outputs'][1]

  def _load_models(self):
    """Load model weights and computation graphs."""
    try:
      self.vision_run_fn = pickle.loads(read_file_chunked(str(self.config.vision_pkl_path)))
      self.policy_run_fn = pickle.loads(read_file_chunked(str(self.config.policy_pkl_path)))
    except (AssertionError, AttributeError, pickle.UnpicklingError, TypeError) as e:
      raise ModelLoadError(
        f"Failed to load pickled model: {e}. "
        "Models need to be recompiled with current tinygrad version. "
        "Run: python selfdrive/modeld/compile_warp.py"
      ) from e

  def _init_state(self):
    """Initialize runner state."""
    # Image queue tensors
    img_queue_shape = (
      6 * (ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ + 1),
      128, 256
    )
    self._img_queues = {
      'img': Tensor.zeros(img_queue_shape, dtype='uint8').contiguous().realize(),
      'big_img': Tensor.zeros(img_queue_shape, dtype='uint8').contiguous().realize()
    }

    # Transform tensors
    self._transforms = {
      k: Tensor.zeros((3, 3), dtype='float32', device='NPY').realize()
      for k in self._img_queues
    }

    # Buffer caches
    self._blob_cache: dict[tuple, Tensor] = {}
    self._full_frames: dict[str, Tensor] = {}
    self._frame_buf_params: dict[str, tuple] = {}

    # Warp function (loaded on first run)
    self._update_imgs_fn = None

    # Policy input state
    self._policy_inputs = {
      k: np.zeros(self.policy_input_shapes[k], dtype=np.float32)
      for k in self.policy_input_shapes
    }

    # Performance tracking
    self._inference_times: list[float] = []
    self._max_history = 100

  def _warmup(self):
    """Perform warmup runs to compile JIT graphs."""
    if DEBUG >= 1:
      print("[TinygradRunner] Starting warmup...")

    # Note: Full warmup requires actual VisionIPC buffers
    # This is typically done in modeld.py main loop on first frame
    # We'll just verify models are loaded
    if DEBUG >= 1:
      print(f"[TinygradRunner] Warmup configured for {self.config.warmup_runs} runs")

  def _prepare_vision_inputs(self,
                             bufs: dict[str, Any],
                             transforms: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """
    Prepare vision model inputs from VisionIPC buffers.

    Args:
      bufs: VisionIPC buffers
      transforms: Camera transformation matrices

    Returns:
      Dictionary of input tensors for vision model
    """
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

    # Update frame parameters if needed
    for key in bufs.keys():
      if key not in self._frame_buf_params:
        w, h = bufs[key].width, bufs[key].height
        self._frame_buf_params[key] = get_nv12_info(w, h)

    # Load warp function if needed
    if self._update_imgs_fn is None:
      first_key = next(iter(bufs.keys()))
      w, h = bufs[first_key].width, bufs[first_key].height
      warp_path = self.config.models_dir / f'warp_{w}x{h}_tinygrad.pkl'
      with open(warp_path, "rb") as f:
        self._update_imgs_fn = pickle.load(f)

    # Prepare frame tensors with zero-copy
    for key in bufs.keys():
      ptr = bufs[key].data.ctypes.data
      yuv_size = self._frame_buf_params[key][3]
      cache_key = (key, ptr)

      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8')
      self._full_frames[key] = self._blob_cache[cache_key]

    # Update transform matrices
    for key in bufs.keys():
      transform_np = self._transforms[key].numpy()
      transform_np[:, :] = transforms[key][:, :]

    # Run image update (warping)
    out = self._update_imgs_fn(
      self._img_queues['img'], self._full_frames['img'], self._transforms['img'],
      self._img_queues['big_img'], self._full_frames['big_img'], self._transforms['big_img']
    )

    # Update image queues
    self._img_queues['img'] = out[0].realize()
    self._img_queues['big_img'] = out[2].realize()

    # Extract warped images for model input
    return {
      'img': out[1],
      'big_img': out[3]
    }

  def run_vision(self,
                 bufs: dict[str, Any],
                 transforms: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run vision model inference.

    Args:
      bufs: VisionIPC input buffers
      transforms: Camera transformation matrices

    Returns:
      Vision model output as numpy array
    """
    if self.profiler:
      self.profiler.start("vision_prepare")

    # Prepare inputs
    vision_inputs = self._prepare_vision_inputs(bufs, transforms)

    if self.profiler:
      self.profiler.end()
      self.profiler.start("vision_execute")

    # Run vision model
    vision_output_tensor = self.vision_run_fn(**vision_inputs)
    vision_output_tensor = vision_output_tensor.contiguous().realize()

    if self.profiler:
      self.profiler.end()
      self.profiler.start("vision_copy")

    # Copy output to CPU
    vision_output = vision_output_tensor.uop.base.buffer.numpy().flatten()

    if self.profiler:
      self.profiler.end()

    return vision_output

  def run_policy(self, policy_inputs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run policy model inference.

    Args:
      policy_inputs: Dictionary of policy input arrays

    Returns:
      Policy model output as numpy array
    """
    if self.profiler:
      self.profiler.start("policy_prepare")

    # Convert numpy inputs to tensors
    tensor_inputs = {
      k: Tensor(v, device='NPY').realize()
      for k, v in policy_inputs.items()
    }

    if self.profiler:
      self.profiler.end()
      self.profiler.start("policy_execute")

    # Run policy model
    policy_output_tensor = self.policy_run_fn(**tensor_inputs)
    policy_output_tensor = policy_output_tensor.contiguous().realize()

    if self.profiler:
      self.profiler.end()
      self.profiler.start("policy_copy")

    # Copy output to CPU
    policy_output = policy_output_tensor.uop.base.buffer.numpy().flatten()

    if self.profiler:
      self.profiler.end()

    return policy_output

  def run(self,
          bufs: dict[str, Any],
          transforms: dict[str, np.ndarray],
          policy_inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Run both vision and policy models.

    Args:
      bufs: VisionIPC input buffers
      transforms: Camera transformation matrices
      policy_inputs: Dictionary of policy input arrays

    Returns:
      Tuple of (vision_output, policy_output)
    """
    start_time = time.perf_counter()

    # Run vision model
    vision_output = self.run_vision(bufs, transforms)

    # Run policy model
    policy_output = self.run_policy(policy_inputs)

    # Track inference time
    inference_time = time.perf_counter() - start_time
    self._inference_times.append(inference_time)
    if len(self._inference_times) > self._max_history:
      self._inference_times.pop(0)

    return vision_output, policy_output

  def get_avg_inference_time(self) -> float:
    """Get average inference time over recent history."""
    if not self._inference_times:
      return 0.0
    return sum(self._inference_times) / len(self._inference_times)

  def check_20hz_compliance(self) -> tuple[bool, float]:
    """
    Check if inference is meeting 20Hz requirement.

    Returns:
      Tuple of (is_compliant, average_inference_time)
    """
    avg_time = self.get_avg_inference_time()
    # Target: inference should complete well within 50ms budget
    # Allow 80% of budget for inference, leaving room for other processing
    target_time = 0.05 * 0.8  # 40ms
    return avg_time <= target_time, avg_time

  def get_stats(self) -> dict[str, Any]:
    """Get comprehensive runner statistics."""
    is_20hz, avg_time = self.check_20hz_compliance()
    stats = {
      'avg_inference_time_ms': avg_time * 1000,
      'last_inference_time_ms': self._inference_times[-1] * 1000 if self._inference_times else 0,
      'meets_20hz_target': is_20hz,
      'target_inference_time_ms': 40.0,
      'gpu_config': {
        'hardware_type': GPU_CONFIG.hardware_type,
        'device': Device.DEFAULT,
      }
    }

    if self.profiler:
      stats['profiler'] = {
        name: self.profiler.get_stats(name)
        for name in self.profiler._profiles.keys()
      }

    return stats

  def print_stats(self):
    """Print statistics to console."""
    stats = self.get_stats()
    print("\n" + "=" * 60)
    print("TinygradRunner Performance Statistics")
    print("=" * 60)
    print(f"Hardware: {stats['gpu_config']['hardware_type']} ({stats['gpu_config']['device']})")
    print(f"Avg inference time: {stats['avg_inference_time_ms']:.2f}ms")
    print(f"Last inference time: {stats['last_inference_time_ms']:.2f}ms")
    print(f"Target: {stats['target_inference_time_ms']:.1f}ms")
    print(f"20Hz compliant: {'✓' if stats['meets_20hz_target'] else '✗'}")

    if self.profiler and 'profiler' in stats:
      print("\nProfiler breakdown:")
      for name, _stats in stats['profiler'].items():
        if _stats:
          print(f"  {name}: {_stats['mean']:.2f}ms (±{_stats['std']:.2f}ms)")

  def clear_caches(self):
    """Clear all caches to free memory."""
    self._blob_cache.clear()
    self._full_frames.clear()
    # Note: Don't clear JIT caches during normal operation
    # as this would require recompilation


# Convenience function for creating runner with standard paths
def create_runner(models_dir: Path | None = None,
                  enable_profiling: bool = False,
                  debug_level: int = 0) -> TinygradRunner:
  """
  Create a TinygradRunner with standard openpilot model paths.

  Args:
    models_dir: Optional models directory
    enable_profiling: Enable performance profiling
    debug_level: Debug level (0=off, 1=basic, 2=verbose)

  Returns:
    Configured TinygradRunner instance
  """
  config = RunnerConfig(
    models_dir=models_dir,
    enable_profiling=enable_profiling,
    debug_level=debug_level,
  )
  return TinygradRunner(config)


# Backward compatibility alias
TinygradModelRunner = TinygradRunner


if __name__ == "__main__":
  # Test runner creation
  print("Testing TinygradRunner creation...")
  runner = create_runner(debug_level=1)
  print("✓ Runner created successfully")
  print(f"  Vision inputs: {runner.vision_input_names}")
  print(f"  Vision output size: {runner.vision_output_size}")
  print(f"  Policy output size: {runner.policy_output_size}")
