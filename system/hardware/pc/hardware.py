"""
PC hardware interface for openpilot development.

This module provides a hardware abstraction layer for development on
standard Linux/Mac/WSL2 systems.
"""
import platform
from cereal import log
from openpilot.system.hardware.base import HardwareBase, DeviceType, ThermalConfig, ThermalZone

NetworkType = log.DeviceState.NetworkType


class Pc(HardwareBase):
  """
  Hardware interface for PC/Mac development systems.
  
  PC hardware provides:
  - Software rendering (no EGL)
  - No real hardware dependencies
  - Standard network interfaces
  """
  
  def __init__(self):
    super().__init__()
    # Set up capabilities for PC
    self._capabilities._has_egl_support = False  # Use software rendering on PC
    self._capabilities._has_gpu_acceleration = platform.system() != "Darwin"
    self._capabilities._requires_realtime = False
    self._capabilities._has_core_affinity_control = platform.system() == "Linux"
    # No specialized hardware on PC
    self._capabilities._has_managed_fan = False
    self._capabilities._has_internal_panda = False
    self._capabilities._has_modem = False
    self._capabilities._has_display = False
    self._capabilities._has_touchscreen = False
    self._capabilities._has_ir_camera = False
    self._capabilities._has_power_monitoring = False
    self._capabilities._has_thermal_zones = False
    self._capabilities._has_screen_brightness_control = False
    self._capabilities._has_power_save_mode = False
    self._capabilities._has_lpa = False
    self._capabilities._has_amplifier = False
  
  def get_device_type(self) -> str:
    return "pc"
  
  def get_device_type_enum(self) -> DeviceType:
    return DeviceType.PC
  
  def get_network_type(self):
    return NetworkType.wifi
  
  def get_thermal_config(self):
    # Return minimal thermal config for PC
    # Try to read from standard Linux thermal zones if available
    cpu_zones = []
    gpu_zones = []
    
    if platform.system() == "Linux":
      # Try to find thermal zones
      import os
      for i in range(10):
        try:
          zone_path = f"/sys/class/thermal/thermal_zone{i}"
          if os.path.exists(zone_path):
            with open(f"{zone_path}/type") as f:
              zone_type = f.read().strip()
              if "cpu" in zone_type.lower() or "x86" in zone_type.lower() or "core" in zone_type.lower():
                cpu_zones.append(ThermalZone(zone_type, zone_number=i))
              elif "gpu" in zone_type.lower() or "i915" in zone_type.lower():
                gpu_zones.append(ThermalZone(zone_type, zone_number=i))
        except (FileNotFoundError, IOError):
          continue
    
    # If no zones found, provide dummy zones
    if not cpu_zones:
      cpu_zones = [ThermalZone("pc_cpu", scale=1.0)]
    if not gpu_zones:
      gpu_zones = [ThermalZone("pc_gpu", scale=1.0)]
    
    return ThermalConfig(cpu=cpu_zones, gpu=gpu_zones)
  
  def get_gpu_usage_percent(self):
    """Get GPU usage percentage if available."""
    if platform.system() == "Darwin":
      # macOS doesn't have easy GPU usage access
      return 0
    
    try:
      # Try to read from Intel GPU if available
      import os
      if os.path.exists("/sys/class/drm/card0/gt_cur_freq_mhz"):
        with open("/sys/class/drm/card0/gt_cur_freq_mhz") as f:
          cur = int(f.read())
        with open("/sys/class/drm/card0/gt_max_freq_mhz") as f:
          max_freq = int(f.read())
        if max_freq > 0:
          return int(100.0 * cur / max_freq)
    except Exception:
      pass
    
    return 0
  
  def booted(self) -> bool:
    return True
