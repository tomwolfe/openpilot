"""
Unified safety model mapping for openpilot.

This module provides a centralized mapping between CarParams.SafetyModel
and the required panda safety configurations, eliminating redundant logic
across car interfaces.
"""
from opendbc.car import structs

# Safety model parameter flags from various car manufacturers
# These are consolidated here for a single source of truth


class HondaSafetyFlags:
  """Honda-specific safety parameter flags."""
  ALT_BRAKE = 1
  NIDEC_ALT = 2
  BOSCH_LONG = 4
  RADARLESS = 8
  BOSCH_CANFD = 16


class HyundaiSafetyFlags:
  """Hyundai-specific safety parameter flags."""
  CANFD = 1
  CAMERA_SCC = 2
  HYBRID = 4
  EV = 8
  LONG = 16
  ALT_LIMITS = 32
  ALT_LIMITS_2 = 64
  FCEV_GAS = 128
  CANFD_LKA_STEERING = 256
  CANFD_ALT_BUTTONS = 512


class ToyotaSafetyFlags:
  """Toyota-specific safety parameter flags."""
  ALT_BRAKE = 1
  SECOC = 2
  LTA = 4
  STOCK_LONGITUDINAL = 8


class VolkswagenSafetyFlags:
  """Volkswagen-specific safety parameter flags."""
  LONG_CONTROL = 1


class ChryslerSafetyFlags:
  """Chrysler-specific safety parameter flags."""
  RAM_HD = 1
  RAM_DT = 2


class GMSafetyFlags:
  """GM-specific safety parameter flags."""
  EV = 1
  HW_CAM = 2
  HW_CAM_LONG = 4


class SubaruSafetyFlags:
  """Subaru-specific safety parameter flags."""
  GEN2 = 1
  LONG = 2
  PREGLOBAL_REVERSED_DRIVER_TORQUE = 4


class NissanSafetyFlags:
  """Nissan-specific safety parameter flags."""
  ALT_EPS_BUS = 1


class RivianSafetyFlags:
  """Rivian-specific safety parameter flags."""
  LONG_CONTROL = 1


# Unified safety model documentation
# Maps each safety model to its description and typical usage
SAFETY_MODEL_INFO = {
  structs.CarParams.SafetyModel.noOutput: {
    "name": "noOutput",
    "description": "No actuator commands sent to panda (passive mode)",
    "typical_use": "Dashcam mode, unsupported vehicles, or initialization"
  },
  structs.CarParams.SafetyModel.silent: {
    "name": "silent",
    "description": "All messages allowed, no controls",
    "typical_use": "Debugging and development"
  },
  structs.CarParams.SafetyModel.allOutput: {
    "name": "allOutput",
    "description": "All actuator commands allowed",
    "typical_use": "Testing and validation"
  },
  structs.CarParams.SafetyModel.hondaNidec: {
    "name": "hondaNidec",
    "description": "Honda Nidec radar/ACC integration",
    "typical_use": "Honda/Acura vehicles with Nidec radar"
  },
  structs.CarParams.SafetyModel.hondaBosch: {
    "name": "hondaBosch",
    "description": "Honda Bosch radar/ACC integration",
    "typical_use": "Honda/Acura vehicles with Bosch radar"
  },
  structs.CarParams.SafetyModel.toyota: {
    "name": "toyota",
    "description": "Toyota LTA/LKA steering and longitudinal control",
    "typical_use": "Toyota/Lexus vehicles with TSS 2.0+"
  },
  structs.CarParams.SafetyModel.hyundai: {
    "name": "hyundai",
    "description": "Hyundai SCC (Smart Cruise Control) integration",
    "typical_use": "Hyundai/Kia/Genesis vehicles with SCC"
  },
  structs.CarParams.SafetyModel.hyundaiCanfd: {
    "name": "hyundaiCanfd",
    "description": "Hyundai CAN-FD protocol support",
    "typical_use": "Newer Hyundai/Kia/Genesis vehicles with CAN-FD"
  },
  structs.CarParams.SafetyModel.hyundaiLegacy: {
    "name": "hyundaiLegacy",
    "description": "Legacy Hyundai SCC without camera-based SCC",
    "typical_use": "Older Hyundai/Kia/Genesis vehicles (pre-2018)"
  },
  structs.CarParams.SafetyModel.volkswagen: {
    "name": "volkswagen",
    "description": "Volkswagen MQB platform support",
    "typical_use": "VW/Audi/Skoda/Seat MQB vehicles"
  },
  structs.CarParams.SafetyModel.volkswagenMlb: {
    "name": "volkswagenMlb",
    "description": "Volkswagen MLB platform support",
    "typical_use": "Audi and premium VW MLB vehicles"
  },
  structs.CarParams.SafetyModel.volkswagenPq: {
    "name": "volkswagenPq",
    "description": "Volkswagen PQ platform support",
    "typical_use": "Older VW/Audi/Skoda/Seat PQ vehicles"
  },
  structs.CarParams.SafetyModel.gm: {
    "name": "gm",
    "description": "GM ASCM/Camera ACC integration",
    "typical_use": "GM vehicles with adaptive cruise control"
  },
  structs.CarParams.SafetyModel.chrysler: {
    "name": "chrysler",
    "description": "Chrysler/Fiat/Ram ACC integration",
    "typical_use": "Chrysler, Dodge, Jeep, Ram, Fiat vehicles"
  },
  structs.CarParams.SafetyModel.subaru: {
    "name": "subaru",
    "description": "Subaru Gen2 Eyesight support",
    "typical_use": "Subaru vehicles with Gen2 Eyesight"
  },
  structs.CarParams.SafetyModel.subaruPreglobal: {
    "name": "subaruPreglobal",
    "description": "Subaru Gen1 Eyesight support",
    "typical_use": "Older Subaru vehicles with Gen1 Eyesight"
  },
  structs.CarParams.SafetyModel.nissan: {
    "name": "nissan",
    "description": "Nissan ProPILOT Assist integration",
    "typical_use": "Nissan/Infiniti vehicles with ProPILOT"
  },
  structs.CarParams.SafetyModel.mazda: {
    "name": "mazda",
    "description": "Mazda FZ-SENSE integration",
    "typical_use": "Mazda vehicles with i-ACTIVSENSE"
  },
  structs.CarParams.SafetyModel.ford: {
    "name": "ford",
    "description": "Ford Co-Pilot360 integration",
    "typical_use": "Ford/Lincoln vehicles with Co-Pilot360"
  },
  structs.CarParams.SafetyModel.tesla: {
    "name": "tesla",
    "description": "Tesla Autopilot integration",
    "typical_use": "Tesla vehicles (development only)"
  },
  structs.CarParams.SafetyModel.psa: {
    "name": "psa",
    "description": "PSA (Peugeot/Citroën/DS/Opel) support",
    "typical_use": "PSA Group vehicles"
  },
  structs.CarParams.SafetyModel.rivian: {
    "name": "rivian",
    "description": "Rivian Driver+ integration",
    "typical_use": "Rivian R1T/R1S vehicles"
  },
  structs.CarParams.SafetyModel.body: {
    "name": "body",
    "description": "Comma body vehicle control",
    "typical_use": "Comma body robotics platform"
  },
}


def get_safety_model_info(safety_model: structs.CarParams.SafetyModel) -> dict:
  """
  Get information about a safety model.

  Args:
    safety_model: The safety model enum value

  Returns:
    Dictionary with name, description, and typical_use keys
  """
  return SAFETY_MODEL_INFO.get(safety_model, {
    "name": "unknown",
    "description": "Unknown safety model",
    "typical_use": "Not documented"
  })


def validate_safety_config(safety_config: structs.CarParams.SafetyConfig) -> bool:
  """
  Validate a safety configuration for consistency.

  Args:
    safety_config: The safety config to validate

  Returns:
    True if valid, raises ValueError if invalid
  """
  if safety_config.safetyModel not in SAFETY_MODEL_INFO:
    raise ValueError(f"Unknown safety model: {safety_config.safetyModel}")

  # Check for common invalid combinations
  if safety_config.safetyModel == structs.CarParams.SafetyModel.noOutput:
    if safety_config.safetyParam != 0:
      raise ValueError("noOutput safety model should have safetyParam=0")

  return True


def create_safety_config(safety_model: structs.CarParams.SafetyModel,
                         safety_param: int = 0) -> structs.CarParams.SafetyConfig:
  """
  Create a validated safety configuration.

  This is a convenience wrapper around get_safety_config that adds validation.

  Args:
    safety_model: The safety model to use
    safety_param: Optional safety parameter flags

  Returns:
    Validated SafetyConfig object
  """
  from opendbc.car import get_safety_config

  config = get_safety_config(safety_model, safety_param if safety_param else None)
  validate_safety_config(config)
  return config
