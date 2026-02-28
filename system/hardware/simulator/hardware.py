"""
Simulator hardware interface for openpilot.

This module provides a hardware abstraction layer for the simulation environment.
The simulator acts as a virtual car interface, feeding CAN packets into the
standardized card process.
"""
import os
from cereal import log
from openpilot.system.hardware.base import HardwareBase, DeviceType, ThermalConfig, ThermalZone

NetworkType = log.DeviceState.NetworkType


class Simulator(HardwareBase):
  """
  Hardware interface for the simulation environment.

  The simulator provides:
  - Virtual CAN interface through the standard card process
  - Simulated sensors (cameras, IMU, GPS)
  - No real hardware dependencies (runs on any Linux/Mac/WSL2)
  """

  def __init__(self):
    super().__init__()
    # Set up capabilities for simulator
    self._capabilities._is_simulator = True
    self._capabilities._has_simulated_car_interface = True
    self._capabilities._has_egl_support = False  # Use software rendering
    self._capabilities._has_gpu_acceleration = False
    self._capabilities._requires_realtime = False
    self._capabilities._has_core_affinity_control = False
    # No physical hardware in simulator
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
    return "simulator"

  def get_device_type_enum(self) -> DeviceType:
    return DeviceType.SIMULATOR

  def get_network_type(self):
    # Simulator uses network for rendering/communication
    return NetworkType.wifi

  def get_network_metered(self, network_type) -> bool:
    # Assume simulator network is not metered
    return False

  def get_thermal_config(self):
    # Return minimal thermal config for simulator
    # In simulation, we don't have real thermal zones
    return ThermalConfig(
      cpu=[ThermalZone("sim_cpu", scale=1.0)],
      gpu=[ThermalZone("sim_gpu", scale=1.0)],
    )

  def get_gpu_usage_percent(self):
    # Return a simulated GPU usage
    try:
      import psutil
      # Use CPU usage as a proxy for simulation load
      return psutil.cpu_percent(interval=0.1)
    except Exception:
      return 0

  def booted(self) -> bool:
    # Simulator is always "booted"
    return True

  def initialize_hardware(self):
    """Initialize simulator hardware (no-op for virtual hardware)."""

  def set_power_save(self, powersave_enabled):
    """Power save mode not applicable in simulation."""

  def get_screen_brightness(self):
    """Return default brightness for simulator UI."""
    return 50

  def set_screen_brightness(self, percentage):
    """Set screen brightness (simulated, no effect)."""

  def set_display_power(self, on: bool):
    """Display power control (simulated, no effect)."""

  def get_sim_info(self):
    """Return simulated SIM info."""
    return {
      'sim_id': 'SIMULATOR',
      'mcc_mnc': None,
      'network_type': ["Simulation"],
      'sim_state': ["NOT_PRESENT"],
      'data_connected': False
    }

  def get_serial(self):
    """Return simulator serial number."""
    return os.environ.get('SIMULATOR_SERIAL', 'SIM00000000')

  def get_voltage(self) -> float:
    """Return simulated battery voltage."""
    return 12.0

  def get_current(self) -> float:
    """Return simulated battery current."""
    return 0.0

  def get_current_power_draw(self):
    """Return simulated power draw."""
    return 0.0

  def get_som_power_draw(self):
    """Return simulated SoM power draw."""
    return 0.0
