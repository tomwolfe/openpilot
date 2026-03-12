#!/usr/bin/env python3
"""
E2E Phase 4: Navigation Embeddings Process

This process converts navigation route data into continuous tensor embeddings
that can be ingested by the driving model for semantic routing.

The embeddings encode:
- Upcoming lane choices (keep left, keep right, exit, fork)
- Distance to maneuvers
- Turn angles and road geometry
- Speed limit changes along route

This allows the E2E model to natively understand navigation intent
without relying on simple blinker-based "desires".
"""
import numpy as np
import cereal.messaging as messaging
from cereal import log
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Priority, Ratekeeper
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

# Embedding dimensions
EMBEDDING_DIM = 64  # Size of navigation embedding vector
MAX_ROUTE_POINTS = 100  # Maximum number of route points to encode


def create_route_embedding(nav_route, nav_instruction, car_state, live_location):
  """
  Create navigation embedding from route and instruction data.

  Args:
    nav_route: Navigation route data (list of coordinates)
    nav_instruction: Current navigation instruction
    car_state: Current vehicle state
    live_location: Current vehicle location

  Returns:
    numpy array of shape (EMBEDDING_DIM,) containing navigation embedding
  """
  embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

  # Early return if no navigation data
  if not nav_route or len(nav_route) == 0:
    return embedding

  try:
    # Extract current position
    if len(live_location) > 0:
      car_lat = live_location[0]
      car_lon = live_location[1]
    else:
      return embedding

    # Encode distance to next maneuver (normalized 0-1, 1 = very close)
    if nav_instruction and hasattr(nav_instruction, 'distance'):
      distance_to_maneuver = nav_instruction.distance
      # Normalize: 0-500m -> 1-0
      embedding[0] = max(0.0, min(1.0, 1.0 - distance_to_maneuver / 500.0))
    else:
      embedding[0] = 0.0  # No upcoming maneuver

    # Encode maneuver type (one-hot encoded in indices 1-10)
    if nav_instruction and hasattr(nav_instruction, 'type'):
      maneuver_type = nav_instruction.type
      # Map maneuver types to indices
      maneuver_map = {
        'turn-left': 1,
        'turn-right': 2,
        'merge-left': 3,
        'merge-right': 4,
        'exit-left': 5,
        'exit-right': 6,
        'fork-left': 7,
        'fork-right': 8,
        'straight': 9,
        'uturn': 10,
      }
      if maneuver_type in maneuver_map:
        embedding[maneuver_map[maneuver_type]] = 1.0

    # Encode route geometry (curvature along path)
    # Sample points along route and encode curvature
    route_points = nav_route[:min(MAX_ROUTE_POINTS, len(nav_route))]
    if len(route_points) > 2:
      # Calculate average curvature over next 100m, 500m, 1km
      for i, distance_bucket in enumerate([10, 50, 100]):
        idx = min(int(distance_bucket * len(route_points) / 1000), len(route_points) - 1)
        if idx > 1:
          # Simple curvature estimate from route points
          embedding[11 + i] = np.clip(route_points[idx].get('curvature', 0.0), -1.0, 1.0)

    # Encode speed limit changes along route
    if nav_instruction and hasattr(nav_instruction, 'speedLimit'):
      speed_limit = nav_instruction.speedLimit
      current_speed = car_state.vEgo * 2.237  # Convert to mph
      if speed_limit > 0:
        # Normalize speed difference
        embedding[14] = np.clip((current_speed - speed_limit) / 30.0, -1.0, 1.0)

    # Encode lane information
    if nav_instruction and hasattr(nav_instruction, 'lane'):
      lane_info = nav_instruction.lane
      # Encode lane position (left=-1, center=0, right=1)
      if 'left' in lane_info.lower():
        embedding[15] = -1.0
      elif 'right' in lane_info.lower():
        embedding[15] = 1.0
      else:
        embedding[15] = 0.0

    # Encode remaining route distance (normalized)
    if hasattr(nav_route, 'totalDistance'):
      total_distance = nav_route.totalDistance
      if total_distance > 0:
        # Assume we track distance traveled
        embedding[16] = 0.5  # Placeholder - would need odometer integration

    # Encode road type (highway, urban, residential)
    if nav_instruction and hasattr(nav_instruction, 'roadType'):
      road_type = nav_instruction.roadType
      road_types = {'highway': 17, 'urban': 18, 'residential': 19, 'rural': 20}
      if road_type in road_types:
        embedding[road_types[road_type]] = 1.0

    # Fill remaining dimensions with route shape encoding
    # This provides a continuous representation of the route geometry
    for i, point in enumerate(route_points[:min(20, len(route_points))]):
      if 'lat' in point and 'lon' in point:
        # Encode relative position (offset from car)
        lat_offset = (point['lat'] - car_lat) * 1000  # Approximate meters
        lon_offset = (point['lon'] - car_lon) * 1000
        embedding[21 + i * 2] = np.clip(lat_offset / 100.0, -1.0, 1.0)
        embedding[21 + i * 2 + 1] = np.clip(lon_offset / 100.0, -1.0, 1.0)

  except Exception as e:
    cloudlog.error(f"nav_embedding: Error creating embedding: {e}")

  return embedding


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  params = Params()
  sm = messaging.SubMaster(['navRoute', 'navInstruction', 'carState', 'liveLocationCalibrated'])
  pm = messaging.PubMaster(['navEmbeddings'])

  rk = Ratekeeper(20, print_delay_threshold=None)

  cloudlog.info("nav_embedding: Starting navigation embedding process")

  last_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

  while True:
    sm.update(10)

    # Only process when navigation is active
    if sm.updated['navInstruction'] or sm.updated['navRoute']:
      nav_route = sm['navRoute'] if sm.valid['navRoute'] else None
      nav_instruction = sm['navInstruction'] if sm.valid['navInstruction'] else None
      car_state = sm['carState'] if sm.valid['carState'] else None
      live_location = sm['liveLocationCalibrated'] if sm.valid['liveLocationCalibrated'] else None

      if car_state is not None and live_location is not None:
        # Extract location data
        if hasattr(live_location, 'geodetic'):
          loc_data = [
            live_location.geodetic.lat,
            live_location.geodetic.lon,
          ]
        else:
          loc_data = []

        # Create embedding
        embedding = create_route_embedding(
          nav_route,
          nav_instruction,
          car_state,
          loc_data
        )

        # Smooth embedding to prevent jerky behavior
        embedding = 0.7 * last_embedding + 0.3 * embedding
        last_embedding = embedding

        # Publish embedding
        dat = messaging.new_message('navEmbeddings')
        dat.valid = sm.all_checks()
        dat.navEmbeddings = embedding.tolist()
        pm.send('navEmbeddings', dat)
    else:
      # Send zero embedding when no navigation
      dat = messaging.new_message('navEmbeddings')
      dat.valid = sm.all_checks()
      dat.navEmbeddings = last_embedding.tolist()
      pm.send('navEmbeddings', dat)

    rk.monitor_time()


if __name__ == "__main__":
  main()
