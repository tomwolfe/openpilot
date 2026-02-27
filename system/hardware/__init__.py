"""
Hardware abstraction layer for openpilot.

This module provides automatic hardware detection and capability-based access.
The HARDWARE object is automatically detected based on the system environment.

For backward compatibility, the following module-level properties are provided:
- TICI: True if running on comma 3X hardware
- AGNOS: True if running on AGNOS (comma 3X OS)
- PC: True if running on a development PC/Mac

New code should use capability checks via HARDWARE.capabilities instead:
- HARDWARE.capabilities.has_managed_fan
- HARDWARE.capabilities.has_internal_panda
- HARDWARE.capabilities.has_egl_support
etc.
"""
import os
import platform
from typing import cast

from openpilot.system.hardware.base import HardwareBase, DeviceType
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.pc.hardware import Pc
from openpilot.system.hardware.simulator.hardware import Simulator


def _detect_hardware() -> HardwareBase:
  """
  Automatically detects and returns the appropriate hardware interface.
  
  Detection order:
  1. Check for TICI file (comma 3X hardware)
  2. Check for SIMULATOR environment variable
  3. Default to PC
  """
  # Check for comma 3X hardware
  if os.path.isfile('/TICI'):
    return Tici()
  
  # Check for simulator environment
  if os.environ.get('SIMULATOR', '0') == '1':
    return Simulator()
  
  # Default to PC
  return Pc()


# Auto-detect hardware
HARDWARE = cast(HardwareBase, _detect_hardware())

# Backward compatibility properties (deprecated - use capabilities instead)
TICI = isinstance(HARDWARE, Tici)
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI and not isinstance(HARDWARE, Simulator)
SIMULATOR = isinstance(HARDWARE, Simulator)

# Export for external use
__all__ = ['HARDWARE', 'TICI', 'AGNOS', 'PC', 'SIMULATOR', 'DeviceType']
