"""
Base class for development hardware (PC and Simulator).

This provides a common base for non-comma hardware that requires
no specialized hardware dependencies or environment variables.
"""
from abc import ABC
from cereal import log
from openpilot.system.hardware.base import HardwareBase, ThermalConfig, ThermalZone

NetworkType = log.DeviceState.NetworkType


class DevelopmentHardware(HardwareBase, ABC):
  """
  Abstract base class for development hardware (PC/Simulator).

  Development hardware provides:
  - No real hardware dependencies
  - Standard network interfaces (wifi/ethernet)
  - Minimal thermal monitoring
  - No modem/cellular capabilities
  """

  def __init__(self):
    super().__init__()
    # Set up common development capabilities
    self._capabilities._requires_realtime = False
    self._capabilities._has_core_affinity_control = False
    # No specialized hardware on development platforms
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

  def get_network_type(self):
    return NetworkType.wifi

  def get_network_metered(self, network_type) -> bool:
    # Development networks typically not metered
    return False

  def get_thermal_config(self):
    # Return minimal thermal config for development
    return ThermalConfig(
      cpu=[ThermalZone("dev_cpu", scale=1.0)],
      gpu=[ThermalZone("dev_gpu", scale=1.0)],
    )

  def booted(self) -> bool:
    # Development hardware is always "booted"
    return True

  def get_sim_info(self):
    """Return development/simulated SIM info."""
    return {
      'sim_id': 'DEVELOPMENT',
      'mcc_mnc': None,
      'network_type': ["Development"],
      'sim_state': ["NOT_PRESENT"],
      'data_connected': False
    }

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
