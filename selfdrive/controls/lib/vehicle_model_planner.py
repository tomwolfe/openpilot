"""
VehicleModelPlanner: Encapsulates all vehicle-specific dynamics planning.

This module provides a clean abstraction between the neural network outputs
and vehicle-specific actuation. The goal is to make openpilot core a pure
"Path Follower" that can be deployed on any vehicle platform.

All vehicle-specific parameters come from CarParams (via opendbc).
"""

import math
import numpy as np

from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants


class VehicleModelPlanner:
  """
  Encapsulates vehicle-specific dynamics for planning.

  This class:
  - Takes model outputs (desired path, velocity, acceleration)
  - Applies vehicle-specific constraints (mass, tire stiffness, etc.)
  - Outputs actuator commands (curvature, acceleration)

  The neural network should be vehicle-agnostic; all car-specific
  logic lives here.
  """

  def __init__(self, CP):
    """
    Initialize with car parameters.

    Args:
      CP: CarParams from opendbc containing all vehicle-specific parameters
    """
    self.CP = CP
    self.VM = VehicleModel(CP)

    # Vehicle-specific limits
    self.accel_min = ACCEL_MIN
    self.accel_max = ACCEL_MAX

    # Turn acceleration limiting lookup table
    self._a_total_max_v = [1.7, 3.2]  # m/s^2
    self._a_total_max_bp = [20., 40.]  # m/s

    # Initialize curvature state
    self.curvature = 0.0

  def compute_desired_curvature(self, model_output, v_ego, roll, prev_curvature,
                                 lat_delay, active=True):
    """
    Compute desired curvature from model output.

    Args:
      model_output: Model V2 output containing path prediction
      v_ego: Ego velocity [m/s]
      roll: Road roll [rad]
      prev_curvature: Previous curvature for smoothing
      lat_delay: Lateral actuator delay [s]
      active: Whether lateral control is active

    Returns:
      desired_curvature: Curvature command [1/m]
      curvature_limited: Whether curvature was limited
    """
    if not active:
      # Return current curvature when not active to avoid jumps
      return self.curvature, False

    # Extract desired curvature from model
    # Model outputs orientation rate, convert to curvature
    # Convert capnp lists to numpy arrays for proper slicing
    yaws = np.array(model_output.orientation.z)
    yaw_rates = np.array(model_output.orientationRate.z)
    desired_curvature = get_curvature_from_plan(
      yaws,
      yaw_rates,
      ModelConstants.T_IDXS,
      v_ego,
      lat_delay
    )

    # Apply vehicle-specific curvature limits
    desired_curvature, curvature_limited = self._clip_curvature(
      v_ego, prev_curvature, desired_curvature, roll
    )

    # Store curvature state for next iteration
    self.curvature = float(desired_curvature)

    return self.curvature, curvature_limited

  def compute_desired_acceleration(self, model_output, v_ego, v_cruise,
                                    longitudinal_active, reset_state,
                                    long_delay, steer_angle, pitch=0.0):
    """
    Compute desired acceleration from model output.

    Args:
      model_output: Model V2 output containing velocity/acceleration prediction
      v_ego: Ego velocity [m/s]
      v_cruise: Cruise setpoint [m/s]
      longitudinal_active: Whether longitudinal control is active
      reset_state: Whether to reset state (user overriding)
      long_delay: Longitudinal actuator delay [s]
      steer_angle: Current steering angle [deg]
      pitch: Road pitch [rad]

    Returns:
      desired_accel: Acceleration command [m/s^2]
      should_stop: Whether vehicle should come to stop
    """
    # Get acceleration from model plan
    # Convert capnp lists to numpy arrays for proper slicing
    velocities = np.array(model_output.velocity.x)
    accelerations = np.array(model_output.acceleration.x)
    desired_accel, should_stop = get_accel_from_plan(
      velocities,
      accelerations,
      ModelConstants.T_IDXS,
      action_t=long_delay
    )

    # Apply vehicle-specific acceleration limits
    accel_clip = [self.accel_min, self._get_max_accel(v_ego)]

    # Limit acceleration in turns to avoid losing traction
    accel_clip = self._limit_accel_in_turns(v_ego, steer_angle, accel_clip)

    # Apply coast acceleration based on pitch
    if pitch != 0.0:
      accel_coast = self._get_coast_accel(pitch)
      accel_clip[1] = min(accel_clip[1], accel_coast)

    # Clip acceleration to vehicle limits
    desired_accel = np.clip(desired_accel, accel_clip[0], accel_clip[1])

    return float(desired_accel), bool(should_stop)

  def update_vehicle_params(self, stiffness_factor, steer_ratio):
    """
    Update vehicle model parameters (e.g., from online estimation).

    Args:
      stiffness_factor: Tire stiffness scaling factor
      steer_ratio: Steering ratio
    """
    self.VM.update_params(stiffness_factor, steer_ratio)

  def get_current_curvature(self, steer_angle, v_ego, roll):
    """
    Get current curvature from vehicle state.

    Args:
      steer_angle: Steering wheel angle [rad]
      v_ego: Ego velocity [m/s]
      roll: Road roll [rad]

    Returns:
      curvature: Current curvature [1/m]
    """
    return -self.VM.calc_curvature(steer_angle, v_ego, roll)

  def _clip_curvature(self, v_ego, prev_curvature, new_curvature, roll):
    """
    Clip curvature to respect ISO lateral jerk and acceleration limits.

    Args:
      v_ego: Ego velocity [m/s]
      prev_curvature: Previous curvature [1/m]
      new_curvature: Requested curvature [1/m]
      roll: Road roll [rad]

    Returns:
      clipped_curvature: Curvature within limits
      limited: Whether curvature was limited
    """
    from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
    return clip_curvature(v_ego, prev_curvature, new_curvature, roll)

  def _get_max_accel(self, v_ego):
    """Get maximum acceleration based on velocity."""
    # Lookup table for max acceleration vs velocity
    a_cruise_max_vals = [1.6, 1.2, 0.8, 0.6]
    a_cruise_max_bp = [0., 10.0, 25., 40.]
    return float(np.interp(v_ego, a_cruise_max_bp, a_cruise_max_vals))

  def _get_coast_accel(self, pitch):
    """Get coast acceleration based on road pitch."""
    # Fitted from real data
    return float(np.sin(pitch) * -5.65 - 0.3)

  def _limit_accel_in_turns(self, v_ego, steer_angle, accel_clip):
    """
    Limit longitudinal acceleration in turns to maintain traction.

    Uses total acceleration circle: sqrt(a_x^2 + a_y^2) <= a_max

    Args:
      v_ego: Ego velocity [m/s]
      steer_angle: Steering wheel angle [deg]
      accel_clip: [min_accel, max_accel]

    Returns:
      accel_clip: Modified acceleration limits
    """
    a_total_max = np.interp(v_ego, self._a_total_max_bp, self._a_total_max_v)

    # Calculate lateral acceleration from steering
    a_y = v_ego ** 2 * steer_angle * CV.DEG_TO_RAD / (self.CP.steerRatio * self.CP.wheelbase)

    # Calculate maximum allowed longitudinal acceleration
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

    return [accel_clip[0], min(accel_clip[1], a_x_allowed)]
