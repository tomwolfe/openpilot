"""
Simulated Car for openpilot simulation.

This module provides the bridge between the simulator world and openpilot's
car interface. It updates the shared simulator state that is read by the
standard card process.

Phase 5: Enhanced for E2E model testing with high-fidelity IMU and visual data.
"""
import traceback
import numpy as np

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState
from openpilot.tools.sim.car_interface.simulator import get_simulator_state


class SimulatedCar:
  """
  Simulates a car for openpilot testing.

  This class translates simulator state into the format expected by openpilot's
  CarInterface. The simulator now acts as a native car interface through the
  standard card process.
  
  Phase 5: Enhanced to provide high-fidelity visual and IMU data for E2E model testing.
  """

  def __init__(self):
    self.params = Params()
    self.simulator_state = get_simulator_state()
    self.idx = 0
    self.obd_multiplexing = False
    self.prev_speed = 0.0
    self.prev_steering_angle = 0.0
    
    # Phase 5: IMU noise parameters for realistic sensor simulation
    self.accel_noise_std = 0.02  # m/s^2
    self.gyro_noise_std = 0.001  # rad/s
    
    # Set initial OBD multiplexing state
    self.obd_multiplexing = self.params.get_bool("ObdMultiplexingEnabled")

  def update(self, simulator_state: SimulatorState):
    """
    Update the shared simulator state from the simulator world.

    This state is read by the standard card process through the
    simulator CarInterface.
    
    Phase 5: Enhanced to provide realistic IMU data for E2E model testing.
    """
    try:
      if not simulator_state.valid:
        return

      # Convert simulator state to CarState format
      speed = simulator_state.speed  # m/s
      steering_angle_rad = simulator_state.steering_angle * 0.02  # Convert to radians
      
      # Phase 5: Calculate realistic acceleration and yaw rate from simulator state
      # This provides high-fidelity IMU data for E2E model
      dt = 0.02  # 50Hz update rate
      a_ego = (speed - self.prev_speed) / dt if dt > 0 else 0.0
      yaw_rate = (steering_angle_rad - self.prev_steering_angle) / dt if dt > 0 else 0.0
      
      # Add realistic IMU noise for E2E model testing
      a_ego += np.random.normal(0, self.accel_noise_std)
      yaw_rate += np.random.normal(0, self.gyro_noise_std)
      
      # Update previous state for next iteration
      self.prev_speed = speed
      self.prev_steering_angle = steering_angle_rad

      # Update shared simulator state with enhanced IMU data
      self.simulator_state.update(
        v_ego=speed,
        v_ego_raw=speed,
        a_ego=a_ego,  # Phase 5: Realistic acceleration from speed changes
        steering_angle=steering_angle_rad,
        steering_rate=yaw_rate,  # Phase 5: Realistic steering rate
        yaw_rate=yaw_rate,  # Phase 5: Yaw rate for E2E model

        gas=simulator_state.user_gas,
        brake=simulator_state.user_brake,
        brake_pressed=simulator_state.user_brake > 0,
        gas_pressed=simulator_state.user_gas > 0,

        cruise_available=True,
        cruise_enabled=simulator_state.is_engaged,
        cruise_set_speed=0.0,  # Set by controls

        steer_torque_sensor=simulator_state.user_torque,
        steer_torque_driver=0.0,
        steer_fault=False,
        steer_warning=False,

        left_blinker=simulator_state.left_blinker,
        right_blinker=simulator_state.right_blinker,

        door_open=False,
        seatbelt_unlatched=False,

        can_valid=True,
        cum_lag_ms=0.0,

        ignition=simulator_state.ignition,
        ignition_can=simulator_state.ignition,
        ignition_line=simulator_state.ignition,

        latitude=simulator_state.gps.latitude,
        longitude=simulator_state.gps.longitude,
        altitude=simulator_state.gps.altitude,
        bearing=simulator_state.bearing,
        
        # Phase 5: Additional fields for E2E model testing
        # IMU data is now provided through camerad/simulated_sensors
      )

      # Handle OBD multiplexing
      if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
        self.obd_multiplexing = not self.obd_multiplexing
        self.params.put_bool("ObdMultiplexingChanged", True)

      self.idx += 1
    except Exception:
      traceback.print_exc()
      raise
