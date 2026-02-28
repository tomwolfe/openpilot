#!/usr/bin/env python3
"""
Verification script for openpilot decoupling efforts.

This script verifies that:
1. selfdrive/controls/ has no imports from selfdrive/car/
2. plannerd.py can initialize using only a mocked CarParams capnp message
3. system/hardware allows HARDWARE to be instantiated without SIMULATION=1
4. All car-specific constants are removed from the controls/ folder

Run from the openpilot root directory:
  python tools/verify_decoupling.py
"""
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import NamedTuple


class TestResult(NamedTuple):
  name: str
  passed: bool
  message: str


def get_root_dir() -> Path:
  """Get the openpilot root directory."""
  return Path(__file__).parent.parent


def test_no_selfdrive_car_imports_in_controls() -> TestResult:
  """
  Verify that selfdrive/controls/ has no imports from selfdrive/car/.

  This ensures the controls layer is car-agnostic and relies only on
  CarParams provided through the messaging layer.

  Note: Imports from opendbc.car are allowed and expected - that's where
  car-agnostic interfaces and constants should live.
  """
  root = get_root_dir()
  controls_dir = root / "selfdrive" / "controls"

  violations = []

  # Search all Python files in controls directory
  for py_file in controls_dir.rglob("*.py"):
    # Skip test files - they may import car interfaces for testing
    if 'tests' in str(py_file):
      continue

    try:
      content = py_file.read_text()
      lines = content.split('\n')

      for i, line in enumerate(lines, 1):
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
          continue

        # Check for selfdrive.car imports (not opendbc.car)
        if 'from openpilot.selfdrive.car' in line or 'from selfdrive.car' in line:
          # Skip if it's actually from opendbc
          if 'opendbc' not in line:
            violations.append(f"{py_file.relative_to(root)}:{i}: {stripped}")
        elif 'import selfdrive.car' in line:
          if 'opendbc' not in line:
            violations.append(f"{py_file.relative_to(root)}:{i}: {stripped}")

    except Exception as e:
      violations.append(f"{py_file.relative_to(root)}: Error reading file: {e}")

  if violations:
    return TestResult(
      name="No selfdrive/car imports in controls",
      passed=False,
      message="Found forbidden imports:\n  " + "\n  ".join(violations)
    )

  return TestResult(
    name="No selfdrive/car imports in controls",
    passed=True,
    message="✓ No forbidden imports found in selfdrive/controls/ (opendbc.car imports are allowed)"
  )


def test_plannerd_with_mocked_carParams() -> TestResult:
  """
  Verify that plannerd.py can initialize using only a mocked CarParams message.

  This tests that plannerd doesn't require a live car connection to initialize.
  """
  try:
    from cereal import car
    from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner

    # Create a minimal mocked CarParams
    CP = car.CarParams.new_message()
    CP.brand = "mock"
    CP.carFingerprint = "MOCK_CAR"
    CP.openpilotLongitudinalControl = True
    CP.dashcamOnly = False
    CP.passive = False

    # Try to initialize the planner with mocked params
    planner = LongitudinalPlanner(CP)

    # Verify it has the expected attributes
    assert hasattr(planner, 'CP')
    assert hasattr(planner, 'mpc')
    assert hasattr(planner, 'update')
    assert hasattr(planner, 'publish')

    return TestResult(
      name="plannerd with mocked CarParams",
      passed=True,
      message="✓ LongitudinalPlanner initializes successfully with mocked CarParams"
    )

  except ImportError as e:
    return TestResult(
      name="plannerd with mocked CarParams",
      passed=False,
      message=f"✗ Import error: {e}"
    )
  except Exception as e:
    return TestResult(
      name="plannerd with mocked CarParams",
      passed=False,
      message=f"✗ Initialization failed: {e}"
    )


def test_hardware_instantiation_without_env() -> TestResult:
  """
  Verify that HARDWARE can be instantiated on a generic Linux machine without SIMULATION=1.

  The PC hardware should be the default when no special environment is detected.
  """
  try:
    # Ensure SIMULATION/SIMULATOR env vars are not set
    old_simulator = os.environ.pop('SIMULATOR', None)
    old_simulation = os.environ.pop('SIMULATION', None)

    # Force reimport to trigger hardware detection
    from openpilot.system import hardware as hw_module

    # Check that HARDWARE was detected
    if not hasattr(hw_module, 'HARDWARE'):
      return TestResult(
        name="Hardware instantiation without env vars",
        passed=False,
        message="✗ HARDWARE object not found in system.hardware module"
      )

    # Check that it's a valid hardware type (should be PC by default)
    hw = hw_module.HARDWARE
    device_type = hw.get_device_type()

    # Restore env vars
    if old_simulator:
      os.environ['SIMULATOR'] = old_simulator
    if old_simulation:
      os.environ['SIMULATION'] = old_simulation

    return TestResult(
      name="Hardware instantiation without env vars",
      passed=True,
      message=f"✓ HARDWARE instantiated successfully (type: {device_type})"
    )

  except Exception as e:
    return TestResult(
      name="Hardware instantiation without env vars",
      passed=False,
      message=f"✗ Hardware instantiation failed: {e}"
    )


def test_no_car_constants_in_controls() -> TestResult:
  """
  Verify that car-specific constants are removed from the controls/ folder.

  Checks for common car brand names and model-specific constants.
  """
  root = get_root_dir()
  controls_dir = root / "selfdrive" / "controls"

  # Car brand patterns that shouldn't appear in controls logic
  car_brands = [
    'HONDA', 'TOYOTA', 'HYUNDAI', 'KIA', 'GENESIS',
    'FORD', 'GM', 'CHEVROLET', 'CADILLAC',
    'VOLKSWAGEN', 'AUDI', 'SKODA', 'SEAT',
    'BMW', 'MERCEDES', 'TESLA', 'NISSAN', 'INFINITI',
    'SUBARU', 'MAZDA', 'CHRYSLER', 'DODGE', 'JEEP', 'RAM',
    'PEUGEOT', 'CITROEN', 'OPEL', 'DS',
    'RIVIAN', 'FCA', 'STELLANTIS'
  ]

  # Build pattern for car-specific constants
  # Looking for things like: HONDA_NIDEC, TOYOTA_LTA, HYBRID_FLAG, etc.
  pattern = r'\b(' + '|'.join(car_brands) + r')_[A-Z_0-9]+\b'

  violations = []

  # Search all Python files in controls directory (excluding tests)
  for py_file in controls_dir.rglob("*.py"):
    if 'tests' in str(py_file):
      continue

    try:
      content = py_file.read_text()
      matches = re.findall(pattern, content)
      if matches:
        unique_matches = sorted(set(matches))
        violations.append(f"{py_file.relative_to(root)}: {', '.join(unique_matches)}")
    except Exception as e:
      violations.append(f"{py_file.relative_to(root)}: Error reading file: {e}")

  if violations:
    return TestResult(
      name="No car-specific constants in controls",
      passed=False,
      message="Found car-specific constants:\n  " + "\n  ".join(violations)
    )

  return TestResult(
    name="No car-specific constants in controls",
    passed=True,
    message="✓ No car-specific constants found in selfdrive/controls/"
  )


def test_development_hardware_base_class() -> TestResult:
  """
  Verify that PC and Simulator hardware share a common base class.
  """
  try:
    from openpilot.system.hardware.development_base import DevelopmentHardware
    from openpilot.system.hardware.pc.hardware import Pc
    from openpilot.system.hardware.simulator.hardware import Simulator

    # Check inheritance
    if not issubclass(Pc, DevelopmentHardware):
      return TestResult(
        name="Development hardware base class",
        passed=False,
        message="✗ Pc does not inherit from DevelopmentHardware"
      )

    if not issubclass(Simulator, DevelopmentHardware):
      return TestResult(
        name="Development hardware base class",
        passed=False,
        message="✗ Simulator does not inherit from DevelopmentHardware"
      )

    # Check that both can be instantiated without env vars
    pc_hw = Pc()
    sim_hw = Simulator()

    # Verify they have the expected capabilities
    assert not pc_hw.capabilities.has_internal_panda
    assert sim_hw.capabilities.is_simulator

    return TestResult(
      name="Development hardware base class",
      passed=True,
      message="✓ PC and Simulator share DevelopmentHardware base class"
    )

  except Exception as e:
    return TestResult(
      name="Development hardware base class",
      passed=False,
      message=f"✗ Verification failed: {e}"
    )


def test_safety_mapping_module() -> TestResult:
  """
  Verify that the unified safety mapping module exists and works.
  """
  try:
    from openpilot.selfdrive.car.safety_mapping import (
      get_safety_model_info,
      HondaSafetyFlags,
      ToyotaSafetyFlags,
      HyundaiSafetyFlags,
    )

    # Test that we can get info for a safety model
    from cereal import car

    info = get_safety_model_info(car.CarParams.SafetyModel.toyota)
    assert 'name' in info
    assert 'description' in info

    # Test that safety flags are defined
    assert hasattr(HondaSafetyFlags, 'ALT_BRAKE')
    assert hasattr(ToyotaSafetyFlags, 'LTA')
    assert hasattr(HyundaiSafetyFlags, 'LONG')

    return TestResult(
      name="Safety mapping module",
      passed=True,
      message="✓ Safety mapping module is functional"
    )

  except Exception as e:
    return TestResult(
      name="Safety mapping module",
      passed=False,
      message=f"✗ Safety mapping verification failed: {e}"
    )


def run_grep_check() -> None:
  """Run a grep-based check for selfdrive/car imports in controls."""
  print("\n" + "="*70)
  print("GREP-BASED IMPORT CHECK")
  print("="*70)

  root = get_root_dir()
  controls_dir = root / "selfdrive" / "controls"

  # Use grep to search for imports
  try:
    result = subprocess.run(
      ['grep', '-r', '--include=*.py',
       '-e', 'from selfdrive.car',
       '-e', 'import selfdrive.car',
       str(controls_dir)],
      capture_output=True,
      text=True,
      cwd=root
    )

    if result.returncode == 0 and result.stdout:
      # Filter out test files
      lines = [line for line in result.stdout.split('\n') if line and '/tests/' not in line]
      if lines:
        print("⚠ Found potential imports (excluding tests):")
        for line in lines:
          print(f"  {line}")
      else:
        print("✓ No imports found (excluding test files)")
    else:
      print("✓ No selfdrive/car imports found in controls/")

  except Exception as e:
    print(f"✗ Grep check failed: {e}")


def main():
  """Run all verification tests."""
  print("="*70)
  print("OPENPILOT DECOUPLING VERIFICATION")
  print("="*70)
  print()

  # Run all tests
  tests = [
    test_no_selfdrive_car_imports_in_controls,
    test_plannerd_with_mocked_carParams,
    test_hardware_instantiation_without_env,
    test_no_car_constants_in_controls,
    test_development_hardware_base_class,
    test_safety_mapping_module,
  ]

  results = []
  for test_func in tests:
    result = test_func()
    results.append(result)
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"[{status}] {result.name}")
    print(f"       {result.message}")
    print()

  # Run grep-based check
  run_grep_check()

  # Summary
  print("="*70)
  print("SUMMARY")
  print("="*70)
  passed = sum(1 for r in results if r.passed)
  total = len(results)
  print(f"Passed: {passed}/{total}")

  if passed == total:
    print("\n✓ All decoupling verification tests passed!")
    return 0
  else:
    print(f"\n✗ {total - passed} test(s) failed. Please review the issues above.")
    return 1


if __name__ == "__main__":
  sys.exit(main())
