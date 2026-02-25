import math
from multiprocessing import Queue
import random

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }

def create_map(track_size=60, randomize=False):
  """Create map configuration. Supports randomization for varied scenarios."""
  if randomize:
    # Generate random track configuration
    num_segments = random.randint(6, 12)
    config = [None]
    for _ in range(num_segments):
      if random.random() > 0.5:
        config.append(straight_block(random.randint(40, 100)))
      else:
        config.append(curve_block(
          random.randint(80, 200),
          random.choice([45, 90, 135]),
          random.choice([0, 1])
        ))
  else:
    # Default oval track
    curve_len = track_size * 2
    config = [
      None,
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
    ]

  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=config
  )


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False,
               photorealistic=False, weather=None, traffic_density=0.0, randomize_map=False):
    super().__init__(dual_camera, high_quality)

    # Enhanced simulation options
    self.should_render = photorealistic  # Enable photorealistic rendering
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf
    self.weather = weather  # None, 'rain', 'fog', 'sunset', 'night'
    self.traffic_density = traffic_density
    self.randomize_map = randomize_map

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    # Enhanced configuration for photorealistic simulation
    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=False,  # Don't render ego vehicle
        image_source="rgb_road",
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=False,
      on_continuous_line_done=False,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      traffic_density=self.traffic_density,  # Configurable traffic
      map_config=create_map(randomize=self.randomize_map),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,
      show_logo=False,
      anisotropic_filtering=False,
      # Enhanced visual settings
      light_mode=self.weather if self.weather else 'daytime',
      skybox_mode=self.weather if self.weather else 'default',
    )

    # Add weather effects if specified
    if self.weather == 'rain':
      config['weather'] = 'rain'
    elif self.weather == 'fog':
      config['weather'] = 'fog'

    # High quality rendering options
    if self.high_quality:
      config.update(dict(
        image_resolution=(W, H),
        image_source="rgb_road",
        stack_size=128,
      ))

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
