"""
Simulator Car Interface for openpilot.

This module provides a CarInterface implementation for the simulation environment.
The simulator acts as a native car interface, feeding CAN packets into the
standardized card process instead of running as a separate bridge daemon.

Usage:
  Set SIMULATOR=1 environment variable and the hardware layer will automatically
  detect the simulator. The car interface will be loaded by the standard card process.
"""
import threading

from opendbc.car import structs
from opendbc.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType
from opendbc.car.can_definitions import CanData, CanRecvCallable, CanSendCallable

from cereal import car
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


# CarState flags
class CarStateFlags:
  pass


class SimulatorCarState:
  """
  Maintains the state of the simulated car.
  This is updated by the simulator world and read by the CarInterface.
  """

  def __init__(self):
    self.lock = threading.Lock()

    # Vehicle dynamics
    self.v_ego = 0.0  # m/s
    self.v_ego_raw = 0.0
    self.a_ego = 0.0
    self.steering_angle = 0.0  # radians
    self.steering_rate = 0.0
    self.yaw_rate = 0.0  # rad/s

    # Pedals
    self.gas = 0.0
    self.brake = 0.0
    self.brake_pressed = False
    self.gas_pressed = False

    # Gear
    self.gear = car.CarState.GearSlot.drive
    self.gear_shifter = car.CarState.GearShifter.drive

    # Cruise control
    self.cruise_available = True
    self.cruise_enabled = False
    self.cruise_set_speed = 0.0  # m/s
    self.cruise_button = 0  # 0=none, 1=up, 2=down, 3=cancel, 4=main

    # Steering
    self.steer_torque_sensor = 0.0
    self.steer_torque_driver = 0.0
    self.steer_fault = False
    self.steer_warning = False
    
    # E2E Phase 2: Direct actuation commands from openpilot
    self.user_torque = 0.0  # Steering torque command [-1, 1]
    self.user_gas = 0.0     # Gas pedal command [0, 1]
    self.user_brake = 0.0   # Brake pedal command [0, 1]
    self.aeb_active = False # AEB override flag

    # Lights
    self.left_blinker = False
    self.right_blinker = False
    self.headlights = False

    # Doors and seatbelt
    self.door_open = False
    self.seatbelt_unlatched = False

    # Errors
    self.errors = []
    self.can_valid = True
    self.cum_lag_ms = 0.0

    # Ignition
    self.ignition = True
    self.ignition_can = True
    self.ignition_line = True

    # GPS/Location
    self.latitude = 0.0
    self.longitude = 0.0
    self.altitude = 0.0
    self.bearing = 0.0

  def update(self, **kwargs):
    """Update state from simulator."""
    with self.lock:
      for key, value in kwargs.items():
        if hasattr(self, key):
          setattr(self, key, value)

  def get_state(self) -> car.CarState:
    """Get current state as CarState message."""
    with self.lock:
      CS = car.CarState.new_message()

      CS.vEgo = self.v_ego
      CS.vEgoRaw = self.v_ego_raw
      CS.aEgo = self.a_ego
      CS.steeringAngle = self.steering_angle
      CS.steeringRate = self.steering_rate
      CS.yawRate = self.yaw_rate

      CS.gas = self.gas
      CS.brake = self.brake
      CS.brakePressed = self.brake_pressed
      CS.gasPressed = self.gas_pressed

      CS.gear = self.gear
      CS.gearShifter = self.gear_shifter

      CS.cruiseState.available = self.cruise_available
      CS.cruiseState.enabled = self.cruise_enabled
      CS.cruiseState.speed = self.cruise_set_speed
      CS.cruiseState.speedCluster = self.cruise_set_speed

      CS.steeringTorque = self.steer_torque_sensor
      CS.steeringTorqueEps = self.steer_torque_driver
      CS.steerFaultPermanent = self.steer_fault
      CS.steerWarning = self.steer_warning
      
      # E2E Phase 2: Expose direct actuation commands
      # These are used by the simulator physics engine
      CS.userTorque = self.user_torque
      CS.userGas = self.user_gas
      CS.userBrake = self.user_brake
      CS.aebActive = self.aeb_active

      CS.leftBlinker = self.left_blinker
      CS.rightBlinker = self.right_blinker

      CS.doorOpen = self.door_open
      CS.seatbeltUnlatched = self.seatbelt_unlatched

      CS.canValid = self.can_valid
      CS.cumLagMs = self.cum_lag_ms

      CS.ignition = self.ignition
      CS.ignitionCan = self.ignition_can
      CS.ignitionLine = self.ignition_line

      CS.gpsLatitude = self.latitude
      CS.gpsLongitude = self.longitude
      CS.gpsAltitude = self.altitude
      CS.gpsBearing = self.bearing

      if self.errors:
        CS.errors = self.errors

      return CS


# Global simulator state - shared between simulator world and CarInterface
_simulator_state: SimulatorCarState | None = None
_simulator_recv_callback: CanRecvCallable | None = None
_simulator_send_callback: CanSendCallable | None = None


def get_simulator_state() -> SimulatorCarState:
  """Get the global simulator state."""
  global _simulator_state
  if _simulator_state is None:
    _simulator_state = SimulatorCarState()
  return _simulator_state


def set_simulator_state(state: SimulatorCarState):
  """Set the global simulator state."""
  global _simulator_state
  _simulator_state = state


class CarInterface(CarInterfaceBase):
  """
  CarInterface implementation for the simulator.

  This class integrates the simulator with the standard openpilot car interface,
  allowing the simulator to feed CAN data through the card process.
  """

  def __init__(self, CP: car.CarParams):
    super().__init__(CP)
    self.CP = CP
    self.simulator_state = get_simulator_state()
    self.params = Params()

    # Frame counter
    self.frame = 0

    cloudlog.info("Simulator CarInterface initialized")

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, target_speed):
    """Return acceleration limits for PID controller."""
    return -4.0, 2.0

  @staticmethod
  def get_non_essential_params(CP: car.CarParams) -> structs.CarParams:
    """Get non-essential parameters for the simulator."""
    return CP

  @staticmethod
  def init(CP: car.CarParams, can_recv: CanRecvCallable, can_send: CanSendCallable,
           fingerprint: dict[int, dict[int, int]], fw_versions: list[bytes] | None = None,
           **kwargs) -> None:
    """Initialize the car interface."""
    global _simulator_recv_callback, _simulator_send_callback
    _simulator_recv_callback = can_recv
    _simulator_send_callback = can_send
    cloudlog.info("Simulator CarInterface init complete")

  def update(self, can_packets: list[list[CanData]]) -> car.CarState:
    """
    Update car state from CAN packets.

    In simulation mode, we primarily use the shared simulator state
    rather than parsing actual CAN messages.
    """
    self.frame += 1

    # Process any incoming CAN packets (for compatibility)
    for packet in can_packets:
      for _msg in packet:
        # Could process actual CAN messages here if needed
        pass

    # Get state from simulator
    CS = self.simulator_state.get_state()

    return CS

  def apply(self, CC: car.CarControl, now_nanos: int | None = None) -> tuple[structs.CarControl.Actuators, list[CanData]]:
    """
    Apply car control commands.

    In simulation, we send control commands back to the simulator world
    via the shared state. Supports both classical control (accel/torque)
    and E2E Phase 2 direct actuation (user_gas/user_brake/user_torque).
    """
    actuators = CC.actuators

    # E2E Phase 2: Check if direct actuation is being used
    # The simulator reads these commands directly for physics simulation
    is_e2e = hasattr(actuators, 'e2eEnabled') and actuators.e2eEnabled
    
    if is_e2e:
      # E2E direct actuation mode
      # Send raw actuator commands to simulator
      self.simulator_state.update(
        user_torque=actuators.torque,
        user_gas=max(0.0, actuators.accel / 2.0) if actuators.accel > 0 else 0.0,
        user_brake=max(0.0, -actuators.accel / 4.0) if actuators.accel < 0 else 0.0,
        aeb_active=actuators.longControlState == car.CarControl.Actuators.LongControlState.off and actuators.accel < -3.0,
        cruise_enabled=CC.cruiseControl.enabled,
      )
    else:
      # Classical control mode
      # Convert accel/curvature to pedal/torque commands for simulator
      gas_cmd = max(0.0, actuators.accel / 2.0) if actuators.accel > 0 else 0.0
      brake_cmd = max(0.0, -actuators.accel / 4.0) if actuators.accel < 0 else 0.0
      
      self.simulator_state.update(
        user_torque=actuators.torque,
        user_gas=gas_cmd,
        user_brake=brake_cmd,
        aeb_active=False,
        cruise_enabled=CC.cruiseControl.enabled,
        cruise_set_speed=CC.cruiseControl.speedOverride if CC.cruiseControl.speedOverride is not None else 0.0,
      )

    # Generate fake CAN messages for compatibility (optional)
    can_sends = []

    return actuators, can_sends

  @staticmethod
  def get_steer_feedforward_v2(desired_angle, v_ego, CP):
    """Get feedforward steering."""
    return 0.0

  def get_steer_feedforward_function(self):
    """Return the feedforward function."""
    return self.get_steer_feedforward_v2

  def torque_from_lateral_accel_siglin(self) -> TorqueFromLateralAccelCallbackType:
    """Return torque from lateral acceleration callback."""
    def callback(sigmoid_val: float, sigmoid_alpha: float = 1.0, sigmoid_k: float = 1.0) -> float:
      return sigmoid_val * 1000.0
    return callback


def get_car_params() -> car.CarParams:
  """Get CarParams for the simulator car."""
  CP = car.CarParams.new_message()

  # Basic car identification
  CP.carName = "simulator"
  CP.carFingerprint = "SIMULATOR"
  CP.carVin = "SIM0000000000000000"

  # Vehicle dimensions and characteristics
  CP.mass = 1700.0  # kg (typical mid-size car)
  CP.wheelbase = 2.7  # meters
  CP.steerRatio = 15.0
  CP.steerRatioRear = 0.0
  CP.centerToFront = CP.wheelbase * 0.4

  # Rotational inertia
  CP.rotationalInertia = 2500.0
  CP.tireStiffnessFactor = 1.0

  # Steering limits
  CP.steerLimitTimer = 0.5
  CP.steerActuatorDelay = 0.1
  CP.steerControlType = car.CarParams.SteerControlType.angle

  # Longitudinal limits
  CP.longitudinalTuning.kiBP = [0.0, 5.0, 20.0]
  CP.longitudinalTuning.kiV = [0.1, 0.05, 0.02]
  CP.longitudinalTuning.kpBP = [0.0, 5.0, 20.0]
  CP.longitudinalTuning.kpV = [0.5, 0.3, 0.1]

  # Lateral tuning
  CP.lateralTuning.pid.kiBP = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
  CP.lateralTuning.pid.kiV = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
  CP.lateralTuning.pid.kpBP = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
  CP.lateralTuning.pid.kpV = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
  CP.lateralTuning.pid.kf = 0.00006
  CP.lateralTuning.pid.posKf = 0.0

  # Cruise control
  CP.minEnableSpeed = 0.0  # Can enable from standstill in simulation
  CP.minSteerSpeed = 0.0
  CP.autoResumeSng = True

  # Safety model
  CP.safetyConfigs = [car.CarParams.SafetyConfig()]
  CP.safetyConfigs[0].safetyModel = car.CarParams.SafetyModel.elm327
  CP.safetyConfigs[0].safetyParam = 0

  # Features
  CP.openpilotLongitudinalControl = True
  CP.enableDm = False  # No driver monitoring in simulation
  CP.enableBsm = False
  CP.enableGasInterceptor = False
  CP.enableAps = False

  # No dashcam mode - full simulation
  CP.dashcamOnly = False
  CP.passive = False

  # No alternative experience
  CP.alternativeExperience = 0

  # No radar
  CP.radarUnavailable = True

  return CP


# Register the simulator car interface
def register_simulator_interface():
  """Register the simulator car interface with opendbc."""
  from opendbc.car.car_helpers import interfaces

  # Import the CarInterface class
  from openpilot.tools.sim.car_interface.simulator import CarInterface

  # Register with opendbc
  interfaces['SIMULATOR'] = type('SimulatorInterfaces', (), {
    'CarInterface': CarInterface,
    'RadarInterface': None,  # No radar in basic simulation
  })

  cloudlog.info("Simulator car interface registered")
