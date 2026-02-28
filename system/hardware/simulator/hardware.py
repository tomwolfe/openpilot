"""
Simulator hardware interface for openpilot.

This module provides a hardware abstraction layer for the simulation environment.
The simulator acts as a virtual car interface, feeding CAN packets into the
standardized card process.
"""
import os
from cereal import log
from openpilot.system.hardware.base import DeviceType
from openpilot.system.hardware.development_base import DevelopmentHardware

NetworkType = log.DeviceState.NetworkType


class Simulator(DevelopmentHardware):
  """
  Hardware interface for the simulation environment.

  The simulator provides:
  - Virtual CAN interface through the standard card process
  - Simulated sensors (cameras, IMU, GPS)
  - No real hardware dependencies (runs on any Linux/Mac/WSL2)
  """

  def __init__(self):
    super().__init__()
    # Set up simulator-specific capabilities
    self._capabilities._is_simulator = True
    self._capabilities._has_simulated_car_interface = True
    self._capabilities._has_egl_support = False  # Use software rendering
    self._capabilities._has_gpu_acceleration = False

  def get_device_type(self) -> str:
    return "simulator"

  def get_device_type_enum(self) -> DeviceType:
    return DeviceType.SIMULATOR

  def get_gpu_usage_percent(self):
    # Return a simulated GPU usage
    try:
      import psutil
      # Use CPU usage as a proxy for simulation load
      return psutil.cpu_percent(interval=0.1)
    except Exception:
      return 0

  def get_serial(self):
    """Return simulator serial number."""
    return os.environ.get('SIMULATOR_SERIAL', 'SIM00000000')
