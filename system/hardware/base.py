import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields
from enum import Enum, auto

from cereal import log

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class DeviceType(Enum):
  """Enum for supported device types."""
  TICI = auto()      # comma 3X
  MICI = auto()      # comma 3X (mici variant)
  PC = auto()        # Development PC/Mac
  SIMULATOR = auto() # Simulation environment


class HardwareCapabilities:
  """Capability flags for hardware features."""

  def __init__(self):
    # Core capabilities
    self._has_managed_fan = False
    self._has_internal_panda = False
    self._has_modem = False
    self._has_display = False
    self._has_touchscreen = False
    self._has_ir_camera = False
    self._has_egl_support = False
    self._has_gpu_acceleration = False
    self._has_power_monitoring = False
    self._has_thermal_zones = False
    self._has_screen_brightness_control = False
    self._has_power_save_mode = False
    self._has_lpa = False
    self._has_amplifier = False

    # System capabilities
    self._requires_realtime = False
    self._has_core_affinity_control = False
    self._is_agnos = False

    # Simulation capabilities
    self._is_simulator = False
    self._has_simulated_car_interface = False

  # Core capabilities
  @property
  def has_managed_fan(self) -> bool:
    return self._has_managed_fan

  @property
  def has_internal_panda(self) -> bool:
    return self._has_internal_panda

  @property
  def has_modem(self) -> bool:
    return self._has_modem

  @property
  def has_display(self) -> bool:
    return self._has_display

  @property
  def has_touchscreen(self) -> bool:
    return self._has_touchscreen

  @property
  def has_ir_camera(self) -> bool:
    return self._has_ir_camera

  @property
  def has_egl_support(self) -> bool:
    return self._has_egl_support

  @property
  def has_gpu_acceleration(self) -> bool:
    return self._has_gpu_acceleration

  @property
  def has_power_monitoring(self) -> bool:
    return self._has_power_monitoring

  @property
  def has_thermal_zones(self) -> bool:
    return self._has_thermal_zones

  @property
  def has_screen_brightness_control(self) -> bool:
    return self._has_screen_brightness_control

  @property
  def has_power_save_mode(self) -> bool:
    return self._has_power_save_mode

  @property
  def has_lpa(self) -> bool:
    return self._has_lpa

  @property
  def has_amplifier(self) -> bool:
    return self._has_amplifier

  # System capabilities
  @property
  def requires_realtime(self) -> bool:
    return self._requires_realtime

  @property
  def has_core_affinity_control(self) -> bool:
    return self._has_core_affinity_control

  @property
  def is_agnos(self) -> bool:
    return self._is_agnos

  # Simulation capabilities
  @property
  def is_simulator(self) -> bool:
    return self._is_simulator

  @property
  def has_simulated_car_interface(self) -> bool:
    return self._has_simulated_car_interface

class LPAError(RuntimeError):
  pass

class LPAProfileNotFoundError(LPAError):
  pass

@dataclass
class Profile:
  iccid: str
  nickname: str
  enabled: bool
  provider: str

@dataclass
class ThermalZone:
  # a zone from /sys/class/thermal/thermal_zone*
  name: str             # a.k.a type
  scale: float = 1000.  # scale to get degrees in C
  zone_number = -1

  def read(self) -> float:
    if self.zone_number < 0:
      for n in os.listdir("/sys/devices/virtual/thermal"):
        if not n.startswith("thermal_zone"):
          continue
        with open(os.path.join("/sys/devices/virtual/thermal", n, "type")) as f:
          if f.read().strip() == self.name:
            self.zone_number = int(n.removeprefix("thermal_zone"))
            break

    try:
      with open(f"/sys/devices/virtual/thermal/thermal_zone{self.zone_number}/temp") as f:
        return int(f.read()) / self.scale
    except FileNotFoundError:
      return 0

@dataclass
class ThermalConfig:
  cpu: list[ThermalZone] | None = None
  gpu: list[ThermalZone] | None = None
  dsp: ThermalZone | None = None
  pmic: list[ThermalZone] | None = None
  memory: ThermalZone | None = None
  intake: ThermalZone | None = None
  exhaust: ThermalZone | None = None
  case: ThermalZone | None = None

  def get_msg(self):
    ret = {}
    for f in fields(ThermalConfig):
      v = getattr(self, f.name)
      if v is not None:
        if isinstance(v, list):
          ret[f.name + "TempC"] = [x.read() for x in v]
        else:
          ret[f.name + "TempC"] = v.read()
    return ret

class LPABase(ABC):
  @abstractmethod
  def list_profiles(self) -> list[Profile]:
    pass

  @abstractmethod
  def get_active_profile(self) -> Profile | None:
    pass

  @abstractmethod
  def delete_profile(self, iccid: str) -> None:
    pass

  @abstractmethod
  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    pass

  @abstractmethod
  def nickname_profile(self, iccid: str, nickname: str) -> None:
    pass

  @abstractmethod
  def switch_profile(self, iccid: str) -> None:
    pass

  def is_comma_profile(self, iccid: str) -> bool:
    return any(iccid.startswith(prefix) for prefix in ('8985235',))

class HardwareBase(ABC):
  def __init__(self):
    self._capabilities = HardwareCapabilities()

  @property
  def capabilities(self) -> HardwareCapabilities:
    """Returns the hardware capabilities object for capability-based checks."""
    return self._capabilities

  @abstractmethod
  def get_device_type(self) -> str:
    """Returns the device type string (e.g., 'tici', 'pc', 'simulator')."""

  def get_device_type_enum(self) -> DeviceType:
    """Returns the device type as an enum. Override in subclasses for specific types."""
    return DeviceType.PC

  @staticmethod
  def get_cmdline() -> dict[str, str]:
    """Reads kernel command line parameters. Returns empty dict if unavailable."""
    try:
      with open('/proc/cmdline') as f:
        cmdline = f.read()
      return {kv[0]: kv[1] for kv in [s.split('=') for s in cmdline.split(' ')] if len(kv) == 2}
    except (OSError, FileNotFoundError):
      return {}

  @staticmethod
  def read_param_file(path, parser, default=0):
    """Safely reads a parameter file, returning default on error."""
    try:
      with open(path) as f:
        return parser(f.read())
    except Exception:
      return default

  def booted(self) -> bool:
    """Returns True if hardware is fully booted and ready."""
    return True

  def reboot(self, reason=None):
    """Reboots the device."""
    print("REBOOT!")

  def uninstall(self):
    """Triggers an uninstall/reset of the system."""
    print("uninstall")

  def get_os_version(self):
    """Returns the OS version string, or None if unavailable."""
    return None

  def get_imei(self, slot) -> str:
    """Returns the IMEI for the given SIM slot."""
    return ""

  def get_serial(self):
    """Returns the device serial number."""
    return ""

  def get_network_info(self):
    """Returns network information dict, or None if unavailable."""
    return None

  def get_network_type(self):
    """Returns the current network type."""
    return NetworkType.none

  def get_sim_info(self):
    """Returns SIM card information."""
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_sim_lpa(self) -> LPABase:
    """Returns the LPA (Local Profile Assistant) for eSIM management."""
    raise NotImplementedError("SIM LPA not available")

  def get_network_strength(self, network_type):
    """Returns the signal strength for the given network type."""
    return NetworkStrength.unknown

  def get_network_metered(self, network_type) -> bool:
    """Returns True if the network is metered."""
    return network_type not in (NetworkType.none, NetworkType.wifi, NetworkType.ethernet)

  def get_current_power_draw(self):
    """Returns current power draw in watts."""
    return 0

  def get_som_power_draw(self):
    """Returns SoM power draw in watts."""
    return 0

  def shutdown(self):
    """Shuts down the device."""
    print("SHUTDOWN!")

  def get_thermal_config(self):
    """Returns thermal configuration for monitoring."""
    return ThermalConfig()

  def set_display_power(self, on: bool):
    """Sets display power on/off."""

  def set_screen_brightness(self, percentage):
    """Sets screen brightness (0-100)."""

  def get_screen_brightness(self):
    """Returns current screen brightness (0-100)."""
    return 0

  def set_power_save(self, powersave_enabled):
    """Enables/disables power save mode."""

  def get_gpu_usage_percent(self):
    """Returns GPU usage percentage."""
    return 0

  def get_modem_version(self):
    """Returns modem firmware version, or None if unavailable."""
    return None

  def get_modem_temperatures(self):
    """Returns list of modem temperatures."""
    return []

  def initialize_hardware(self):
    """Initializes hardware components. Called once at startup."""

  def configure_modem(self):
    """Configures modem settings."""

  def reboot_modem(self):
    """Reboots the cellular modem."""

  def get_networks(self):
    """Returns network scan results."""
    return None

  def has_internal_panda(self) -> bool:
    """Returns True if device has an internal panda (CAN interface)."""
    return self._capabilities.has_internal_panda

  def reset_internal_panda(self):
    """Resets the internal panda."""

  def recover_internal_panda(self):
    """Recovers the internal panda from a bad state."""

  def get_modem_data_usage(self):
    """Returns (tx_bytes, rx_bytes) for modem data usage."""
    return -1, -1

  def get_voltage(self) -> float:
    """Returns battery voltage."""
    return 0.

  def get_current(self) -> float:
    """Returns battery current."""
    return 0.

  def set_ir_power(self, percent: int):
    """Sets IR emitter power (0-100)."""
