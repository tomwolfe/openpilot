#!/usr/bin/env python3
"""
E2E Longitudinal Benchmarking Script

This script processes a provided route and calculates a "Comfort Score" by measuring
the variance between the model's raw acceleration prediction and the actual MPC output.
It outputs a comparison table between "Chill Mode" (Heuristic) and "Experimental Mode" (E2E).

Usage:
  python benchmark_e2e.py <route> [--mode experimental|chill]

Example:
  python benchmark_e2e.py "02c45f73a2e5c6e9|2021-01-01--19-08-22" --mode experimental
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from cereal import messaging
from openpilot.common.constants import CV
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentName


@dataclass
class ComfortMetrics:
  """Metrics for evaluating longitudinal comfort."""
  # Jerk metrics
  mean_jerk: float = 0.0
  std_jerk: float = 0.0
  max_jerk: float = 0.0
  jerk_95th: float = 0.0
  
  # Acceleration tracking error (model vs MPC)
  mean_accel_error: float = 0.0
  std_accel_error: float = 0.0
  max_accel_error: float = 0.0
  accel_error_95th: float = 0.0
  
  # Velocity tracking error
  mean_velocity_error: float = 0.0
  std_velocity_error: float = 0.0
  
  # Comfort score (0-100, higher is better)
  comfort_score: float = 0.0
  
  # Additional statistics
  total_samples: int = 0
  experimental_mode_active: float = 0.0  # Percentage of time in experimental mode
  e2e_valid: float = 0.0  # Percentage of time E2E trajectory is valid


@dataclass
class SegmentData:
  """Data collected from a route segment."""
  timestamps: list[float] = field(default_factory=list)
  v_ego: list[float] = field(default_factory=list)
  a_ego: list[float] = field(default_factory=list)
  a_mpc: list[float] = field(default_factory=list)
  a_model: list[float] = field(default_factory=list)
  j_mpc: list[float] = field(default_factory=list)
  experimental_mode: list[bool] = field(default_factory=list)
  e2e_valid: list[bool] = field(default_factory=list)
  lead_status: list[bool] = field(default_factory=list)
  radar_unavailable: list[bool] = field(default_factory=list)


def compute_jerk(accelerations: np.ndarray, dt: float = 0.02) -> np.ndarray:
  """Compute jerk from acceleration samples."""
  if len(accelerations) < 2:
    return np.zeros_like(accelerations)
  jerk = np.diff(accelerations) / dt
  return np.concatenate([[0.0], jerk])


def calculate_comfort_score(metrics: ComfortMetrics) -> float:
  """
  Calculate a comfort score from 0-100 based on various metrics.
  
  The score is penalized by:
  - High jerk (especially peaks)
  - Large acceleration tracking errors
  - High variance in velocity tracking
  """
  score = 100.0
  
  # Jerk penalty (target: < 2 m/s^3 for comfort)
  jerk_penalty = min(metrics.mean_jerk * 5.0, 30.0)  # Max 30 points
  jerk_peak_penalty = min(metrics.max_jerk * 2.0, 20.0)  # Max 20 points
  
  # Acceleration error penalty (target: < 0.5 m/s^2)
  accel_error_penalty = min(metrics.mean_accel_error * 20.0, 25.0)  # Max 25 points
  accel_error_peak_penalty = min(metrics.max_accel_error * 5.0, 15.0)  # Max 15 points
  
  # Velocity error penalty
  velocity_error_penalty = min(metrics.mean_velocity_error * 10.0, 10.0)  # Max 10 points
  
  score -= (jerk_penalty + jerk_peak_penalty + accel_error_penalty + 
            accel_error_peak_penalty + velocity_error_penalty)
  
  return max(0.0, min(100.0, score))


def process_route(route: str, mode_filter: Optional[str] = None) -> SegmentData:
  """
  Process a route and extract longitudinal data.
  
  Args:
    route: Route identifier (e.g., "02c45f73a2e5c6e9|2021-01-01--19-08-22")
    mode_filter: Filter by mode ("experimental" or "chill"), None for all
  
  Returns:
    SegmentData with collected metrics
  """
  data = SegmentData()
  
  try:
    lr = LogReader(route)
  except Exception as e:
    print(f"Error loading route {route}: {e}")
    return data
  
  prev_timestamp = 0.0
  
  for msg in lr:
    if msg.which() == 'selfdriveState':
      sd = msg.selfdriveState
      current_time = msg.logMonoTime / 1e9
      
      # Apply mode filter if specified
      if mode_filter == "experimental" and not sd.experimentalMode:
        continue
      elif mode_filter == "chill" and sd.experimentalMode:
        continue
      
      data.timestamps.append(current_time)
      data.experimental_mode.append(sd.experimentalMode)
      
    elif msg.which() == 'carState':
      cs = msg.carState
      data.v_ego.append(cs.vEgo)
      data.a_ego.append(cs.aEgo)
      
    elif msg.which() == 'longitudinalPlan':
      lp = msg.longitudinalPlan
      data.a_mpc.append(lp.aTarget if len(lp.accels) > 0 else 0.0)
      data.e2e_valid.append(lp.modelConfidence > 0.5 if hasattr(lp, 'modelConfidence') else False)
      
      # Get jerk from plan
      if len(lp.jerks) > 0:
        data.j_mpc.append(lp.jerks[0])
      else:
        data.j_mpc.append(0.0)
      
    elif msg.which() == 'modelV2':
      model = msg.modelV2
      # Get model's desired acceleration
      if hasattr(model.action, 'desiredAcceleration'):
        data.a_model.append(model.action.desiredAcceleration)
      else:
        data.a_model.append(0.0)
      
    elif msg.which() == 'radarState':
      rs = msg.radarState
      data.lead_status.append(rs.leadOne.status)
      # Radar unavailable if no leads detected
      data.radar_unavailable.append(not rs.leadOne.status and not rs.leadTwo.status)
  
  return data


def compute_metrics(data: SegmentData) -> ComfortMetrics:
  """Compute comfort metrics from collected data."""
  metrics = ComfortMetrics()
  
  if len(data.a_mpc) == 0 or len(data.a_model) == 0:
    return metrics
  
  # Convert to numpy arrays
  a_mpc = np.array(data.a_mpc)
  a_model = np.array(data.a_model)
  j_mpc = np.array(data.j_mpc) if len(data.j_mpc) > 0 else compute_jerk(a_mpc)
  v_ego = np.array(data.v_ego) if len(data.v_ego) > 0 else np.zeros(len(a_mpc))
  
  # Ensure arrays have same length
  min_len = min(len(a_mpc), len(a_model), len(j_mpc), len(v_ego))
  a_mpc = a_mpc[:min_len]
  a_model = a_model[:min_len]
  j_mpc = j_mpc[:min_len]
  v_ego = v_ego[:min_len]
  
  # Jerk metrics
  jerk_abs = np.abs(j_mpc)
  metrics.mean_jerk = float(np.mean(jerk_abs))
  metrics.std_jerk = float(np.std(jerk_abs))
  metrics.max_jerk = float(np.max(jerk_abs))
  metrics.jerk_95th = float(np.percentile(jerk_abs, 95))
  
  # Acceleration tracking error (model vs MPC)
  accel_error = np.abs(a_model - a_mpc)
  metrics.mean_accel_error = float(np.mean(accel_error))
  metrics.std_accel_error = float(np.std(accel_error))
  metrics.max_accel_error = float(np.max(accel_error))
  metrics.accel_error_95th = float(np.percentile(accel_error, 95))
  
  # Velocity tracking (ideal velocity from model integration)
  # For simplicity, use ego velocity variance as proxy
  if len(v_ego) > 1:
    v_diff = np.diff(v_ego)
    metrics.mean_velocity_error = float(np.mean(np.abs(v_diff)))
    metrics.std_velocity_error = float(np.std(np.abs(v_diff)))
  
  # Experimental mode statistics
  if len(data.experimental_mode) > 0:
    metrics.experimental_mode_active = float(np.mean(data.experimental_mode)) * 100.0
  
  if len(data.e2e_valid) > 0:
    metrics.e2e_valid = float(np.mean(data.e2e_valid)) * 100.0
  
  metrics.total_samples = min_len
  
  # Calculate overall comfort score
  metrics.comfort_score = calculate_comfort_score(metrics)
  
  return metrics


def print_comparison_table(experimental_metrics: ComfortMetrics, 
                          chill_metrics: ComfortMetrics) -> None:
  """Print a comparison table between experimental and chill modes."""
  print("\n" + "="*80)
  print("E2E LONGITUDINAL BENCHMARK COMPARISON")
  print("="*80)
  
  print(f"\n{'Metric':<35} {'Experimental (E2E)':<20} {'Chill (Heuristic)':<20} {'Difference':<15}")
  print("-"*80)
  
  def format_row(name: str, exp_val: float, chill_val: float, suffix: str = "") -> None:
    diff = exp_val - chill_val
    diff_str = f"{diff:+.3f}{suffix}" if diff != 0 else f"{diff:.3f}{suffix}"
    print(f"{name:<35} {exp_val:>10.3f}{suffix:<10} {chill_val:>10.3f}{suffix:<10} {diff_str:>15}")
  
  # Jerk metrics
  print("\n📊 JERK METRICS (m/s³)")
  print("-"*80)
  format_row("Mean Jerk", experimental_metrics.mean_jerk, chill_metrics.mean_jerk)
  format_row("Std Jerk", experimental_metrics.std_jerk, chill_metrics.std_jerk)
  format_row("Max Jerk", experimental_metrics.max_jerk, chill_metrics.max_jerk)
  format_row("95th Percentile Jerk", experimental_metrics.jerk_95th, chill_metrics.jerk_95th)
  
  # Acceleration tracking
  print("\n📈 ACCELERATION TRACKING ERROR (m/s²)")
  print("-"*80)
  format_row("Mean Error", experimental_metrics.mean_accel_error, chill_metrics.mean_accel_error)
  format_row("Std Error", experimental_metrics.std_accel_error, chill_metrics.std_accel_error)
  format_row("Max Error", experimental_metrics.max_accel_error, chill_metrics.max_accel_error)
  format_row("95th Percentile Error", experimental_metrics.accel_error_95th, chill_metrics.accel_error_95th)
  
  # Velocity tracking
  print("\n📉 VELOCITY VARIABILITY (m/s)")
  print("-"*80)
  format_row("Mean Velocity Change", experimental_metrics.mean_velocity_error, chill_metrics.mean_velocity_error)
  format_row("Std Velocity Change", experimental_metrics.std_velocity_error, chill_metrics.std_velocity_error)
  
  # Mode statistics
  print("\n🔧 MODE STATISTICS")
  print("-"*80)
  format_row("Experimental Mode Active", experimental_metrics.experimental_mode_active, 
             chill_metrics.experimental_mode_active, "%")
  format_row("E2E Valid", experimental_metrics.e2e_valid, chill_metrics.e2e_valid, "%")
  print(f"{'Total Samples':<35} {experimental_metrics.total_samples:>10d}{'':<10} {chill_metrics.total_samples:>10d}{'':<10}")
  
  # Overall score
  print("\n" + "="*80)
  print(f"{'OVERALL COMFORT SCORE (0-100)':<35} {experimental_metrics.comfort_score:>10.1f}{'':<10} {chill_metrics.comfort_score:>10.1f}{'':<10} {experimental_metrics.comfort_score - chill_metrics.comfort_score:>+10.1f}")
  print("="*80)
  
  # Recommendation
  if experimental_metrics.comfort_score > chill_metrics.comfort_score:
    print("\n✅ Experimental (E2E) mode shows IMPROVED comfort over Chill mode")
  elif experimental_metrics.comfort_score < chill_metrics.comfort_score:
    print("\n⚠️  Experimental (E2E) mode shows REDUCED comfort compared to Chill mode")
  else:
    print("\nℹ️  Experimental (E2E) and Chill modes show SIMILAR comfort levels")
  
  print()


def main():
  parser = argparse.ArgumentParser(description="Benchmark E2E longitudinal control performance")
  parser.add_argument("route", type=str, help="Route identifier to analyze")
  parser.add_argument("--mode", type=str, choices=["experimental", "chill", "both"], 
                      default="both", help="Filter by driving mode")
  parser.add_argument("--output", type=str, help="Output file for results (optional)")
  
  args = parser.parse_args()
  
  print(f"Processing route: {args.route}")
  print(f"Mode filter: {args.mode}")
  
  # Process route for both modes
  if args.mode in ["experimental", "both"]:
    print("\nProcessing Experimental (E2E) mode data...")
    exp_data = process_route(args.route, mode_filter="experimental")
    exp_metrics = compute_metrics(exp_data)
  else:
    exp_metrics = ComfortMetrics()
  
  if args.mode in ["chill", "both"]:
    print("Processing Chill (Heuristic) mode data...")
    chill_data = process_route(args.route, mode_filter="chill")
    chill_metrics = compute_metrics(chill_data)
  else:
    chill_metrics = ComfortMetrics()
  
  # Print comparison
  if args.mode == "both":
    print_comparison_table(exp_metrics, chill_metrics)
  elif args.mode == "experimental":
    print_comparison_table(exp_metrics, exp_metrics)
  else:
    print_comparison_table(chill_metrics, chill_metrics)
  
  # Save to file if requested
  if args.output:
    with open(args.output, 'w') as f:
      f.write(f"Route: {args.route}\n")
      f.write(f"Mode: {args.mode}\n")
      f.write(f"Experimental Comfort Score: {exp_metrics.comfort_score:.2f}\n")
      f.write(f"Chill Comfort Score: {chill_metrics.comfort_score:.2f}\n")
      f.write(f"Difference: {exp_metrics.comfort_score - chill_metrics.comfort_score:+.2f}\n")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
  main()
