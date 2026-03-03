import numpy as np
import time

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.sim.lib.common import W, H


# Phase 5: E2E model timing constants
# The E2E modeld expects 20fps (50ms frame time) for proper closed-loop operation
E2E_FRAME_RATE = 20  # Hz
E2E_FRAME_TIME = 1.0 / E2E_FRAME_RATE  # 0.05 seconds


def rgb_to_nv12(rgb):
  """Convert RGB image to NV12 (YUV420) format using BT.601 coefficients."""
  h, w = rgb.shape[:2]
  r = rgb[:, :, 0].astype(np.int32)
  g = rgb[:, :, 1].astype(np.int32)
  b = rgb[:, :, 2].astype(np.int32)

  # Y plane - BT.601 coefficients (matches original OpenCL kernel)
  y = (((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16
  y = np.clip(y, 0, 255).astype(np.uint8)

  # Subsample RGB for UV (2x2 box filter)
  r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2

  # U and V planes
  u = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)

  # Interleave UV for NV12 format
  uv = np.empty((h // 2, w), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v

  return np.concatenate([y.ravel(), uv.ravel()]).tobytes()


class Camerad:
  """
  Simulates the camerad daemon.
  
  Phase 5: Enhanced to provide proper VIPC timing for E2E model.
  The E2E modeld expects 20fps (50ms frame time) for proper closed-loop operation.
  """
  def __init__(self, dual_camera):
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    # Phase 5: Increase buffer count for E2E model stability
    # E2E model needs consistent frame delivery without drops
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 8, W, H)
    if dual_camera:
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 8, W, H)

    self.vipc_server.start_listener()
    
    # Phase 5: Track timing for E2E frame rate enforcement
    self.last_frame_time = time.monotonic()

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def rgb_to_yuv(self, rgb):
    """Convert RGB to NV12 YUV format."""
    assert rgb.shape == (H, W, 3), f"{rgb.shape}"
    assert rgb.dtype == np.uint8
    return rgb_to_nv12(rgb)

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    # Phase 5: Enforce E2E frame rate timing
    # This ensures the model receives frames at the expected 20fps rate
    current_time = time.monotonic()
    elapsed = current_time - self.last_frame_time
    
    # Wait if we're sending frames too fast (maintain 20fps)
    if elapsed < E2E_FRAME_TIME:
      time.sleep(E2E_FRAME_TIME - elapsed)
    
    # Update timing for next frame
    self.last_frame_time = time.monotonic()
    eof = int(self.last_frame_time * 1e9)  # Use actual timestamp in nanoseconds
    
    # Phase 5: Send frame with proper timing metadata for E2E model
    # The E2E model uses timestampSof and timestampEof for temporal alignment
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0],
      # Phase 5: Add timing metadata for E2E model
      "timestampSof": eof,
      "timestampEof": eof,
      "frameDropPerc": 0.0,  # No frame drops in simulation
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)
