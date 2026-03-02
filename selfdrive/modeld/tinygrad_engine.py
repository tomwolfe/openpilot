#!/usr/bin/env python3
"""
tinygrad Neural Engine for openpilot Phase 1 E2E 1.0

This module provides optimized neural network inference using tinygrad with:
- Big Graph scheduling for reduced GPU-CPU overhead
- TinyJit kernel fusion for 20Hz real-time performance
- AMD GPU backend targeting TICI hardware
- Zero-copy buffer management for VisionIPC integration
- Schedule cache reuse across frames
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import Device, Buffer
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.schedule import schedule_cache
from tinygrad.helpers import Context, DEBUG, getenv
from tinygrad.nn.state import safe_load, torch_load
from tinygrad.dtype import dtypes

from openpilot.common.file_chunker import read_file_chunked
from openpilot.selfdrive.modeld.constants import ModelConstants


class ModelLoadError(Exception):
  """Raised when model files cannot be loaded (e.g., version mismatch)."""
  pass


# Configuration for TICI AMD GPU
USBGPU = getenv("USBGPU", 0)
if USBGPU:
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'
elif Device.DEFAULT == "AMD" or getenv("QCOM"):
  os.environ['DEV'] = 'AMD'

# Performance tuning constants
INFERENCE_BATCH_SIZE = 1  # Single frame inference
WARMUP_RUNS = 3  # Number of warmup runs for JIT compilation
SCHEDULE_CACHE_SIZE = 16  # Maximum schedule cache entries


class BigGraphScheduler:
  """
  Implements "One-Shot" realization strategy for the entire driving model.
  
  Instead of realizing the graph at every layer, this schedules the entire
  model as one large UOp graph to reduce GPU-CPU synchronization overhead.
  
  References:
  - tinygrad_repo/docs/abstractions3.py
  - tinygrad_repo/test/unit/test_schedule_cache.py
  """
  
  def __init__(self, model_run_fn, prune: bool = True):
    """
    Initialize the Big Graph scheduler with TinyJit.
    
    Args:
      model_run_fn: The model inference function to JIT compile
      prune: Whether to prune unused operations during JIT
    """
    self.jit_runner = TinyJit(model_run_fn, prune=prune)
    self._warmup_complete = False
    self._last_var_vals = {}
    
  def warmup(self, *example_inputs: Tensor):
    """
    Perform warmup runs to compile the JIT graph and populate schedule cache.
    
    Args:
      example_inputs: Sample input tensors matching expected input shapes
    """
    if self._warmup_complete:
      return
    
    with Context(TRACK_MATCH_STATS=0):
      for i in range(WARMUP_RUNS):
        _ = self.jit_runner(*example_inputs)
        Device.default.synchronize()
    
    self._warmup_complete = True
    if DEBUG >= 1:
      print(f"[BigGraphScheduler] Warmup complete, schedule cache size: {len(schedule_cache)}")
  
  def __call__(self, *inputs: Tensor, var_vals: dict[str, int] | None = None) -> list[Tensor]:
    """
    Execute the model with Big Graph scheduling.
    
    Args:
      inputs: Input tensors for the model
      var_vals: Optional variable bindings for symbolic shapes
    
    Returns:
      List of output tensors
    """
    # Ensure warmup is complete
    if not self._warmup_complete:
      self.warmup(*inputs)
    
    # Execute JIT-compiled graph
    outputs = self.jit_runner(*inputs)
    
    # Ensure outputs are realized but not copied back to CPU yet
    # This allows for batched realization when multiple outputs are needed
    return outputs
  
  def clear_cache(self):
    """Clear the schedule cache to free memory."""
    schedule_cache.clear()
    self._warmup_complete = False
  
  @property
  def cache_size(self) -> int:
    """Return current schedule cache size."""
    return len(schedule_cache)


class ZeroCopyBufferManager:
  """
  Manages zero-copy buffer transfers between VisionIPC and tinygrad.
  
  This minimizes latency by:
  1. Using Tensor.from_blob() for direct memory access
  2. Maintaining a buffer cache to avoid reallocation
  3. Supporting NV12 image format directly
  """
  
  def __init__(self, buffer_shapes: dict[str, tuple], dtype: str = 'uint8'):
    """
    Initialize buffer manager.
    
    Args:
      buffer_shapes: Dictionary mapping buffer names to their shapes
      dtype: Data type for buffers (default: uint8 for images)
    """
    self.buffer_shapes = buffer_shapes
    self.dtype = dtype
    self._buffer_cache: dict[tuple, Tensor] = {}
    self._max_cache_size = 64  # Maximum cached buffers
  
  def get_blob_tensor(self, ptr: int, shape: tuple, cache_key: tuple | None = None) -> Tensor:
    """
    Get or create a Tensor from a memory blob.
    
    Args:
      ptr: Memory pointer (ctypes.data)
      shape: Tensor shape
      cache_key: Optional cache key for buffer reuse
    
    Returns:
      Tensor wrapping the memory blob
    """
    if cache_key is not None and cache_key in self._buffer_cache:
      return self._buffer_cache[cache_key]
    
    tensor = Tensor.from_blob(ptr, shape, dtype=self.dtype)
    
    if cache_key is not None:
      # Cache management with LRU eviction
      if len(self._buffer_cache) >= self._max_cache_size:
        # Remove oldest entry
        oldest_key = next(iter(self._buffer_cache))
        del self._buffer_cache[oldest_key]
      self._buffer_cache[cache_key] = tensor
    
    return tensor
  
  def clear_cache(self):
    """Clear all cached buffers."""
    self._buffer_cache.clear()
  
  def prepare_image_buffer(self, buf: Any, frame_params: dict) -> Tensor:
    """
    Prepare an image buffer from VisionIPC for model input.
    
    Args:
      buf: VisionIPC buffer with .data attribute
      frame_params: Frame parameters from get_nv12_info
    
    Returns:
      Tensor ready for model processing
    """
    ptr = buf.data.ctypes.data
    yuv_size = frame_params[3]
    cache_key = (id(buf), ptr)
    
    return self.get_blob_tensor(ptr, (yuv_size,), cache_key=cache_key)


class TinygradModelRunner:
  """
  Main model inference runner using tinygrad backend.
  
  This class orchestrates:
  - Model loading from pickle/safetensors
  - Big Graph scheduling
  - Buffer management
  - Output serialization for cereal
  """
  
  def __init__(self, 
               vision_pkl_path: Path,
               policy_pkl_path: Path,
               vision_metadata_path: Path,
               policy_metadata_path: Path,
               models_dir: Path):
    """
    Initialize the tinygrad model runner.
    
    Args:
      vision_pkl_path: Path to vision model pickle
      policy_pkl_path: Path to policy model pickle
      vision_metadata_path: Path to vision model metadata
      policy_metadata_path: Path to policy model metadata
      models_dir: Directory containing model files
    """
    self.models_dir = models_dir
    self._load_metadata(vision_metadata_path, policy_metadata_path)
    self._load_models(vision_pkl_path, policy_pkl_path)
    
    # Initialize buffer manager
    self.buffer_manager = ZeroCopyBufferManager(
      buffer_shapes=self.vision_input_shapes,
      dtype='uint8'
    )
    
    # Initialize Big Graph schedulers
    self.vision_scheduler = BigGraphScheduler(self._run_vision_raw, prune=True)
    self.policy_scheduler = BigGraphScheduler(self._run_policy_raw, prune=True)
    
    # State tracking
    self._blob_cache: dict[tuple, Tensor] = {}
    self._transforms: dict[str, Tensor] = {}
    self._img_queues: dict[str, Tensor] = {}
    self._full_frames: dict[str, Tensor] = {}
    self._update_imgs_fn = None
    self._frame_buf_params: dict[str, tuple] = {}
    
    # Initialize image queues
    self._init_img_queues()
  
  def _load_metadata(self, vision_path: Path, policy_path: Path):
    """Load model metadata."""
    with open(vision_path, 'rb') as f:
      vision_meta = pickle.load(f)
      self.vision_input_shapes = vision_meta['input_shapes']
      self.vision_input_names = list(self.vision_input_shapes.keys())
      self.vision_output_slices = vision_meta['output_slices']
      self.vision_output_size = vision_meta['output_shapes']['outputs'][1]
    
    with open(policy_path, 'rb') as f:
      policy_meta = pickle.load(f)
      self.policy_input_shapes = policy_meta['input_shapes']
      self.policy_output_slices = policy_meta['output_slices']
      self.policy_output_size = policy_meta['output_shapes']['outputs'][1]
  
  def _load_models(self, vision_pkl: Path, policy_pkl: Path):
    """Load model weights and computation graphs."""
    try:
      self.vision_run_fn = pickle.loads(read_file_chunked(str(vision_pkl)))
      self.policy_run_fn = pickle.loads(read_file_chunked(str(policy_pkl)))
    except (AssertionError, AttributeError, pickle.UnpicklingError, TypeError) as e:
      # Handle version mismatch in pickled JIT objects
      # This can happen when tinygrad version changes
      raise ModelLoadError(
        f"Failed to load pickled model: {e}. "
        f"Models need to be recompiled with current tinygrad version. "
        f"Run: python selfdrive/modeld/compile_warp.py"
      ) from e
  
  def _init_img_queues(self):
    """Initialize image queue tensors."""
    img_queue_shape = (
      6 * (ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ + 1),
      128, 256
    )
    
    self._img_queues = {
      'img': Tensor.zeros(img_queue_shape, dtype='uint8').contiguous().realize(),
      'big_img': Tensor.zeros(img_queue_shape, dtype='uint8').contiguous().realize()
    }
    
    # Initialize transform tensors
    self._transforms = {
      k: Tensor.zeros((3, 3), dtype='float32', device='NPY').realize()
      for k in self._img_queues
    }
  
  def _run_vision_raw(self, **kwargs) -> list[Tensor]:
    """Raw vision model execution (wrapped by BigGraphScheduler)."""
    return [self.vision_run_fn(**kwargs)]
  
  def _run_policy_raw(self, **kwargs) -> list[Tensor]:
    """Raw policy model execution (wrapped by BigGraphScheduler)."""
    return [self.policy_run_fn(**kwargs)]
  
  def _prepare_inputs(self, 
                      bufs: dict[str, Any],
                      transforms: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """
    Prepare model inputs from VisionIPC buffers.
    
    Args:
      bufs: VisionIPC buffers
      transforms: Camera transformation matrices
    
    Returns:
      Dictionary of input tensors
    """
    # Update frame parameters if needed
    for key in bufs.keys():
      if key not in self._frame_buf_params:
        from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
        w, h = bufs[key].width, bufs[key].height
        self._frame_buf_params[key] = get_nv12_info(w, h)
    
    # Load or update warp function if needed
    if self._update_imgs_fn is None:
      # Get dimensions from first buffer
      first_key = next(iter(bufs.keys()))
      w, h = bufs[first_key].width, bufs[first_key].height
      warp_path = self.models_dir / f'warp_{w}x{h}_tinygrad.pkl'
      with open(warp_path, "rb") as f:
        self._update_imgs_fn = pickle.load(f)
    
    # Prepare frame tensors with zero-copy
    for key in bufs.keys():
      ptr = bufs[key].data.ctypes.data
      yuv_size = self._frame_buf_params[key][3]
      cache_key = (key, ptr)
      
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = self.buffer_manager.get_blob_tensor(
          ptr, (yuv_size,), cache_key
        )
      self._full_frames[key] = self._blob_cache[cache_key]
    
    # Update transform matrices
    for key in bufs.keys():
      # Update NPY tensor with new transform
      transform_np = self._transforms[key].numpy()
      transform_np[:, :] = transforms[key][:, :]
    
    return self._full_frames
  
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
    # Prepare inputs
    frames = self._prepare_inputs(bufs, transforms)
    
    # Run image update (warping)
    out = self._update_imgs_fn(
      self._img_queues['img'], self._full_frames['img'], self._transforms['img'],
      self._img_queues['big_img'], self._full_frames['big_img'], self._transforms['big_img']
    )
    
    # Update image queues
    self._img_queues['img'] = out[0].realize()
    self._img_queues['big_img'] = out[2].realize()
    
    # Extract warped images for model input
    vision_inputs = {
      'img': out[1],
      'big_img': out[3]
    }
    
    # Run vision model with Big Graph scheduling
    outputs = self.vision_scheduler(**vision_inputs)
    
    # Realize and copy output to CPU
    vision_output = outputs[0].contiguous().realize()
    return vision_output.uop.base.buffer.numpy().flatten()
  
  def run_policy(self, policy_inputs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run policy model inference.
    
    Args:
      policy_inputs: Dictionary of policy input arrays
    
    Returns:
      Policy model output as numpy array
    """
    # Convert numpy inputs to tensors
    tensor_inputs = {
      k: Tensor(v, device='NPY').realize()
      for k, v in policy_inputs.items()
    }
    
    # Run policy model with Big Graph scheduling
    outputs = self.policy_scheduler(**tensor_inputs)
    
    # Realize and copy output to CPU
    policy_output = outputs[0].contiguous().realize()
    return policy_output.uop.base.buffer.numpy().flatten()
  
  def get_performance_stats(self) -> dict[str, Any]:
    """
    Get performance statistics for debugging and profiling.
    
    Returns:
      Dictionary of performance metrics
    """
    return {
      'vision_schedule_cache_size': self.vision_scheduler.cache_size,
      'policy_schedule_cache_size': self.policy_scheduler.cache_size,
      'blob_cache_size': len(self._blob_cache),
      'buffer_cache_size': len(self.buffer_manager._buffer_cache),
    }
  
  def clear_caches(self):
    """Clear all caches to free memory."""
    self.vision_scheduler.clear_cache()
    self.policy_scheduler.clear_cache()
    self.buffer_manager.clear_cache()
    self._blob_cache.clear()


class TinygradEngine:
  """
  High-level tinygrad inference engine for openpilot modeld.
  
  This provides a drop-in replacement for the legacy inference engine
  with optimized tinygrad backend targeting TICI AMD GPU.
  
  Usage:
    engine = TinygradEngine(vision_pkl, policy_pkl, vision_meta, policy_meta, models_dir)
    
    # For each frame:
    vision_output = engine.infer_vision(bufs, transforms)
    policy_output = engine.infer_policy(policy_inputs)
  """
  
  def __init__(self,
               vision_pkl_path: Path,
               policy_pkl_path: Path,
               vision_metadata_path: Path,
               policy_metadata_path: Path,
               models_dir: Path):
    """
    Initialize the tinygrad engine.
    
    Args:
      vision_pkl_path: Path to vision model pickle
      policy_pkl_path: Path to policy model pickle
      vision_metadata_path: Path to vision model metadata
      policy_metadata_path: Path to policy model metadata
      models_dir: Directory containing model files and warp pickles
    """
    self.model_runner = TinygradModelRunner(
      vision_pkl_path=vision_pkl_path,
      policy_pkl_path=policy_pkl_path,
      vision_metadata_path=vision_metadata_path,
      policy_metadata_path=policy_metadata_path,
      models_dir=models_dir
    )
    
    # Policy input state
    self.numpy_inputs = {
      k: np.zeros(self.model_runner.policy_input_shapes[k], dtype=np.float32)
      for k in self.model_runner.policy_input_shapes
    }
    
    # Performance tracking
    self._last_inference_time = 0.0
    self._inference_times: list[float] = []
    self._max_history = 100
  
  def infer_vision(self,
                   bufs: dict[str, Any],
                   transforms: dict[str, np.ndarray],
                   measure_time: bool = True) -> np.ndarray:
    """
    Run vision model inference.
    
    Args:
      bufs: VisionIPC input buffers
      transforms: Camera transformation matrices
      measure_time: Whether to measure inference time
    
    Returns:
      Vision model output as numpy array
    """
    start_time = time.perf_counter() if measure_time else 0
    
    output = self.model_runner.run_vision(bufs, transforms)
    
    if measure_time:
      self._last_inference_time = time.perf_counter() - start_time
      self._inference_times.append(self._last_inference_time)
      if len(self._inference_times) > self._max_history:
        self._inference_times.pop(0)
    
    return output
  
  def infer_policy(self,
                   policy_inputs: dict[str, np.ndarray],
                   measure_time: bool = True) -> np.ndarray:
    """
    Run policy model inference.
    
    Args:
      policy_inputs: Dictionary of policy input arrays
      measure_time: Whether to measure inference time
    
    Returns:
      Policy model output as numpy array
    """
    start_time = time.perf_counter() if measure_time else 0
    
    output = self.model_runner.run_policy(policy_inputs)
    
    if measure_time:
      self._last_inference_time = time.perf_counter() - start_time
      self._inference_times.append(self._last_inference_time)
      if len(self._inference_times) > self._max_history:
        self._inference_times.pop(0)
    
    return output
  
  def get_avg_inference_time(self) -> float:
    """Get average inference time over recent history."""
    if not self._inference_times:
      return 0.0
    return sum(self._inference_times) / len(self._inference_times)
  
  def check_20hz_compliance(self) -> tuple[bool, float]:
    """
    Check if inference is meeting 20Hz requirement (DT_MDL = 0.05s).
    
    Returns:
      Tuple of (is_compliant, average_inference_time)
    """
    avg_time = self.get_avg_inference_time()
    # Target: inference should complete well within 50ms budget
    # Allow 80% of budget for inference, leaving room for other processing
    target_time = 0.05 * 0.8  # 40ms
    return avg_time <= target_time, avg_time
  
  def get_stats(self) -> dict[str, Any]:
    """Get comprehensive engine statistics."""
    is_20hz, avg_time = self.check_20hz_compliance()
    return {
      'avg_inference_time_ms': avg_time * 1000,
      'last_inference_time_ms': self._last_inference_time * 1000,
      'meets_20hz_target': is_20hz,
      'target_inference_time_ms': 40.0,
      **self.model_runner.get_performance_stats()
    }
  
  def warmup(self, example_bufs: dict[str, Any], example_transforms: dict[str, np.ndarray]):
    """
    Perform warmup to compile JIT graphs.
    
    Args:
      example_bufs: Example VisionIPC buffers
      example_transforms: Example transformation matrices
    """
    print("[TinygradEngine] Starting warmup...")
    
    # Run vision warmup
    _ = self.infer_vision(example_bufs, example_transforms, measure_time=False)
    
    # Run policy warmup
    example_policy_inputs = {
      k: np.zeros(self.model_runner.policy_input_shapes[k], dtype=np.float32)
      for k in self.model_runner.policy_input_shapes
    }
    _ = self.infer_policy(example_policy_inputs, measure_time=False)
    
    print(f"[TinygradEngine] Warmup complete. Stats: {self.get_stats()}")


# Convenience function for creating engine from standard paths
def create_tinygrad_engine(models_dir: Path | None = None) -> TinygradEngine:
  """
  Create a TinygradEngine with standard openpilot model paths.
  
  Args:
    models_dir: Optional models directory (defaults to selfdrive/modeld/models)
  
  Returns:
    Configured TinygradEngine instance
  """
  if models_dir is None:
    models_dir = Path(__file__).parent / 'models'
  
  return TinygradEngine(
    vision_pkl_path=models_dir / 'driving_vision_tinygrad.pkl',
    policy_pkl_path=models_dir / 'driving_policy_tinygrad.pkl',
    vision_metadata_path=models_dir / 'driving_vision_metadata.pkl',
    policy_metadata_path=models_dir / 'driving_policy_metadata.pkl',
    models_dir=models_dir
  )
