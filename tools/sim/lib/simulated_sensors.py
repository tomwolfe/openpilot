import time
import numpy as np

from cereal import log
import cereal.messaging as messaging

from openpilot.common.realtime import DT_DMON
from openpilot.tools.sim.lib.camerad import Camerad

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from openpilot.tools.sim.lib.common import World, SimulatorState


class SimulatedSensors:
  """Simulates the C3 sensors (acc, gyro, gps, peripherals, dm state, cameras) to OpenPilot.
  
  Provides realistic, noisy sensor data to simulate real-world conditions during SIL tests.
  """

  def __init__(self, dual_camera=False):
    self.pm = messaging.PubMaster(['accelerometer', 'gyroscope', 'gpsLocationExternal', 'driverStateV2', 'driverMonitoringState', 'peripheralState'])
    self.camerad = Camerad(dual_camera=dual_camera)
    self.last_perp_update = 0
    self.last_dmon_update = 0
    
    # Noise parameters for realistic sensor simulation
    # IMU noise characteristics (based on typical MEMS IMU specs)
    self.imu_accel_noise_std = 0.05  # m/s^2
    self.imu_gyro_noise_std = 0.01   # rad/s
    self.imu_accel_bias = np.array([0.0, 0.0, 9.81])  # Gravity bias on z-axis
    self.imu_gyro_bias = np.array([0.0, 0.0, 0.0])
    
    # GPS noise characteristics
    self.gps_lat_lon_noise_std = 5e-7    # ~5cm noise
    self.gps_alt_noise_std = 0.1         # 10cm vertical noise
    self.gps_speed_noise_std = 0.05      # m/s
    self.gps_bearing_noise_std = 0.5     # degrees
    
    # Random state for reproducible noise
    self._rng = np.random.default_rng(seed=42)

  def _add_noise(self, value: float | np.ndarray, std: float) -> float | np.ndarray:
    """Add Gaussian noise to a value."""
    if isinstance(value, np.ndarray):
      return value + self._rng.normal(0, std, size=value.shape)
    return value + self._rng.normal(0, std)

  def send_imu_message(self, simulator_state: 'SimulatorState'):
    for _ in range(5):
      # Add realistic noise to accelerometer readings
      accel_raw = np.array([
        simulator_state.imu.accelerometer.x,
        simulator_state.imu.accelerometer.y,
        simulator_state.imu.accelerometer.z
      ])
      accel_noisy = self._add_noise(accel_raw, self.imu_accel_noise_std) + self.imu_accel_bias
      
      dat = messaging.new_message('accelerometer', valid=True)
      dat.accelerometer.sensor = 4
      dat.accelerometer.type = 0x10
      dat.accelerometer.timestamp = dat.logMonoTime
      dat.accelerometer.init('acceleration')
      dat.accelerometer.acceleration.v = [float(accel_noisy[0]), float(accel_noisy[1]), float(accel_noisy[2])]
      self.pm.send('accelerometer', dat)

      # Add realistic noise to gyroscope readings
      gyro_raw = np.array([
        simulator_state.imu.gyroscope.x,
        simulator_state.imu.gyroscope.y,
        simulator_state.imu.gyroscope.z
      ])
      gyro_noisy = self._add_noise(gyro_raw, self.imu_gyro_noise_std) + self.imu_gyro_bias
      
      # copied these numbers from locationd
      dat = messaging.new_message('gyroscope', valid=True)
      dat.gyroscope.sensor = 5
      dat.gyroscope.type = 0x10
      dat.gyroscope.timestamp = dat.logMonoTime
      dat.gyroscope.init('gyroUncalibrated')
      dat.gyroscope.gyroUncalibrated.v = [float(gyro_noisy[0]), float(gyro_noisy[1]), float(gyro_noisy[2])]
      self.pm.send('gyroscope', dat)

  def send_gps_message(self, simulator_state: 'SimulatorState'):
    if not simulator_state.valid:
      return

    # transform from vel to NED
    velNED = [
      -simulator_state.velocity.y,
      simulator_state.velocity.x,
      simulator_state.velocity.z,
    ]
    
    # Add realistic noise to GPS readings
    lat_noisy = self._add_noise(simulator_state.gps.latitude, self.gps_lat_lon_noise_std)
    lon_noisy = self._add_noise(simulator_state.gps.longitude, self.gps_lat_lon_noise_std)
    alt_noisy = self._add_noise(simulator_state.gps.altitude, self.gps_alt_noise_std)
    speed_noisy = max(0.0, self._add_noise(simulator_state.speed, self.gps_speed_noise_std))
    bearing_noisy = self._add_noise(simulator_state.imu.bearing, self.gps_bearing_noise_std) % 360

    for _ in range(10):
      dat = messaging.new_message('gpsLocationExternal', valid=True)
      dat.gpsLocationExternal = {
        "unixTimestampMillis": int(time.time() * 1000),  # noqa: TID251
        "flags": 1,  # valid fix
        "horizontalAccuracy": 1.0,
        "verticalAccuracy": 1.0,
        "speedAccuracy": 0.1,
        "bearingAccuracyDeg": 0.1,
        "vNED": velNED,
        "bearingDeg": float(bearing_noisy),
        "latitude": float(lat_noisy),
        "longitude": float(lon_noisy),
        "altitude": float(alt_noisy),
        "speed": float(speed_noisy),
        "source": log.GpsLocationData.SensorSource.ublox,
      }

      self.pm.send('gpsLocationExternal', dat)

  def send_peripheral_state(self):
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    self.pm.send('peripheralState', dat)

  def send_fake_driver_monitoring(self):
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
    self.pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState', valid=True)
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    self.pm.send('driverMonitoringState', dat)

  def send_camera_images(self, world: 'World'):
    world.image_lock.acquire()
    yuv = self.camerad.rgb_to_yuv(world.road_image)
    self.camerad.cam_send_yuv_road(yuv)

    if world.dual_camera:
      yuv = self.camerad.rgb_to_yuv(world.wide_road_image)
      self.camerad.cam_send_yuv_wide_road(yuv)

  def update(self, simulator_state: 'SimulatorState', world: 'World'):
    now = time.monotonic()
    self.send_imu_message(simulator_state)
    self.send_gps_message(simulator_state)

    if (now - self.last_dmon_update) > DT_DMON/2:
      self.send_fake_driver_monitoring()
      self.last_dmon_update = now

    if (now - self.last_perp_update) > 0.25:
      self.send_peripheral_state()
      self.last_perp_update = now
