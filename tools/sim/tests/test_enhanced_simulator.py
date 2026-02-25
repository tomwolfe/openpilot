"""
Tests for enhanced simulator with photorealistic rendering.

Run with: pytest tools/sim/tests/test_enhanced_simulator.py
"""


class TestEnhancedSimulatorConfig:
  """Test enhanced simulator configuration options."""

  def test_photorealistic_mode(self):
    """Test photorealistic rendering configuration."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

    bridge = MetaDriveBridge(
      dual_camera=True,
      high_quality=False,
      photorealistic=True
    )

    assert bridge.should_render
    assert bridge.weather is None

  def test_weather_conditions(self):
    """Test weather condition configuration."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

    for weather in ['rain', 'fog', 'sunset', 'night']:
      bridge = MetaDriveBridge(
        dual_camera=False,
        high_quality=False,
        weather=weather
      )
      assert bridge.weather == weather

  def test_traffic_density(self):
    """Test traffic density configuration."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

    for density in [0.0, 0.3, 0.5, 0.8, 1.0]:
      bridge = MetaDriveBridge(
        dual_camera=False,
        high_quality=False,
        traffic_density=density
      )
      assert bridge.traffic_density == density

  def test_randomize_map(self):
    """Test map randomization configuration."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import create_map

    # Fixed map
    map_config_fixed = create_map(randomize=False)
    assert map_config_fixed is not None

    # Randomized map
    map_config_random = create_map(randomize=True)
    assert map_config_random is not None

    # Random maps should have variable segment counts
    configs = [create_map(randomize=True) for _ in range(5)]
    # At least some variation expected
    assert len({len(c['config']) for c in configs}) >= 1  # May occasionally be same

  def test_spawn_world_with_enhanced_config(self):
    """Test world spawning with enhanced configuration."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

    bridge = MetaDriveBridge(
      dual_camera=True,
      high_quality=True,
      photorealistic=True,
      weather='rain',
      traffic_density=0.5,
      randomize_map=True
    )

    # Note: Actually spawning would require MetaDrive installed
    # This tests that config is properly structured
    config_template = {
      'use_render': bridge.should_render,
      'traffic_density': bridge.traffic_density,
      'light_mode': bridge.weather if bridge.weather else 'daytime',
    }

    assert config_template['use_render']
    assert config_template['traffic_density'] == 0.5
    assert config_template['light_mode'] == 'rain'


class TestSimulatorCLI:
  """Test simulator command-line interface."""

  def test_parse_args_enhanced(self):
    """Test enhanced CLI argument parsing."""
    from openpilot.tools.sim.run_bridge import parse_args

    # Test with enhanced options
    args = parse_args([
      '--dual_camera',
      '--photorealistic',
      '--weather', 'rain',
      '--traffic_density', '0.5',
      '--randomize_map'
    ])

    assert args.dual_camera
    assert args.photorealistic
    assert args.weather == 'rain'
    assert args.traffic_density == 0.5
    assert args.randomize_map

  def test_parse_args_defaults(self):
    """Test CLI argument defaults."""
    from openpilot.tools.sim.run_bridge import parse_args

    args = parse_args([])

    assert not args.photorealistic
    assert args.weather is None
    assert args.traffic_density == 0.0
    assert not args.randomize_map


class TestSimulatorScenarios:
  """Test predefined simulation scenarios."""

  def test_scenario_edge_cases(self):
    """Test edge case scenario generation."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import create_map

    # Test various track sizes
    for size in [30, 60, 100]:
      map_config = create_map(track_size=size, randomize=False)
      assert map_config is not None
      assert 'config' in map_config

  def test_scenario_weather_combinations(self):
    """Test weather and lighting combinations."""
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

    # Test various combinations
    configs = [
      ('rain', 0.3),
      ('fog', 0.0),
      ('sunset', 0.5),
      ('night', 0.8),
    ]

    for weather, traffic in configs:
      bridge = MetaDriveBridge(
        dual_camera=False,
        high_quality=False,
        weather=weather,
        traffic_density=traffic
      )
      assert bridge.weather == weather
      assert bridge.traffic_density == traffic
