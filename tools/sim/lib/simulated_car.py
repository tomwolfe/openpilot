"""
Simulated Car for openpilot simulation.

This module provides the bridge between the simulator world and openpilot's
car interface. It updates the shared simulator state that is read by the
standard card process.
"""
import traceback

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState
from openpilot.tools.sim.car_interface.simulator import get_simulator_state


class SimulatedCar:
  """
  Simulates a car for openpilot testing.

  This class translates simulator state into the format expected by openpilot's
  CarInterface. The simulator now acts as a native car interface through the
  standard card process.
  """

  def __init__(self):
    self.params = Params()
    self.simulator_state = get_simulator_state()
    self.idx = 0
    self.obd_multiplexing = False

    # Set initial OBD multiplexing state
    self.obd_multiplexing = self.params.get_bool("ObdMultiplexingEnabled")

  def update(self, simulator_state: SimulatorState):
    """
    Update the shared simulator state from the simulator world.

    This state is read by the standard card process through the
    simulator CarInterface.
    """
    try:
      if not simulator_state.valid:
        return

      # Convert simulator state to CarState format
      speed = simulator_state.speed  # m/s
      steering_angle_rad = simulator_state.steering_angle * 0.02  # Convert to radians

      # Update shared simulator state
      self.simulator_state.update(
        v_ego=speed,
        v_ego_raw=speed,
        a_ego=0.0,  # Could be calculated from speed changes
        steering_angle=steering_angle_rad,
        steering_rate=0.0,
        yaw_rate=0.0,

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
      )

      # Handle OBD multiplexing
      if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
        self.obd_multiplexing = not self.obd_multiplexing
        self.params.put_bool("ObdMultiplexingChanged", True)

      self.idx += 1
    except Exception:
      traceback.print_exc()
      raise
