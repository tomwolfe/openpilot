import math
from multiprocessing import Queue

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

def create_map(track_size=60):
  curve_len = track_size * 2
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
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
  )

# Phase 5: Edge-case scenarios for E2E testing
def create_stop_sign_scenario(track_size=60):
  """Scenario with stop signs to test E2E stop/go behavior."""
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      {"id": "stop_sign", "stop_sign": True},  # Stop sign
      straight_block(track_size),
    ],
    # Enable traffic lights and signs
    traffic_light_prob=0.3,
  )

def create_changing_light_scenario(track_size=60):
  """Scenario with traffic lights to test E2E light response."""
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      {"id": "traffic_light", "traffic_light": True},  # Traffic light
      straight_block(track_size),
    ],
    traffic_light_prob=0.5,
  )

def create_merge_scenario(track_size=60):
  """Scenario with merging lanes to test E2E lane change behavior."""
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=3,  # Extra lane for merging
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      {"id": "merge", "merge": True},  # Merge point
      straight_block(track_size),
    ],
  )

def create_cut_in_scenario(track_size=60):
  """Scenario with potential cut-in vehicles to test E2E emergency response."""
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      curve_block(track_size, 45),
      straight_block(track_size),
    ],
    # Add some traffic for cut-in scenarios
    traffic_density=0.1,
  )


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False, scenario="default"):
    """
    Initialize MetaDrive bridge.
    
    Args:
      dual_camera: Use both road and wide cameras
      high_quality: High quality rendering
      test_duration: Duration of test in seconds
      test_run: Whether this is a test run
      scenario: Scenario type ("default", "stop_sign", "changing_light", "merge", "cut_in")
    """
    super().__init__(dual_camera, high_quality)

    self.should_render = False
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf
    self.scenario = scenario  # Phase 5: Scenario selection for E2E testing

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    # Phase 5: Select map based on scenario
    if self.scenario == "stop_sign":
      map_config = create_stop_sign_scenario()
    elif self.scenario == "changing_light":
      map_config = create_changing_light_scenario()
    elif self.scenario == "merge":
      map_config = create_merge_scenario()
    elif self.scenario == "cut_in":
      map_config = create_cut_in_scenario()
    else:
      map_config = create_map()

    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=False,
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
      traffic_density=map_config.get("traffic_density", 0.0),
      map_config=map_config,
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,
      show_logo=False,
      anisotropic_filtering=False,
      # Phase 5: Enable traffic lights and signs for E2E testing
      traffic_light_prob=map_config.get("traffic_light_prob", 0.0),
    )

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
