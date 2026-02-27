#pragma once

#include <string>

#include "system/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_name() { return "pc"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }
  static bool PC() { return true; }
  static bool TICI() { return false; }
  static bool AGNOS() { return false; }
  // Check both SIMULATION (legacy, used by process_replay) and SIMULATOR
  static bool SIMULATOR() { return util::getenv("SIMULATOR", 0) == 1 || util::getenv("SIMULATION", 0) == 1; }
};
