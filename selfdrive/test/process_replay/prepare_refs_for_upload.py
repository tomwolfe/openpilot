#!/usr/bin/env python3
"""
Script to prepare process replay reference logs for upload to ci-artifacts fork.

Usage:
  1. Fork commaai/ci-artifacts to your GitHub account
  2. Run this script to generate reference logs
  3. Upload the generated files to your fork
  4. Update BASE_URL in test_processes.py to point to your fork
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

OPENPILOT_ROOT = Path(__file__).parent.parent.parent.parent
PROC_REPLAY_DIR = OPENPILOT_ROOT / "selfdrive" / "test" / "process_replay"
FAKEDATA_DIR = PROC_REPLAY_DIR / "fakedata"
OUTPUT_DIR = PROC_REPLAY_DIR / "ref_upload"


def run_command(cmd, check=True):
  """Run a shell command."""
  print(f"Running: {cmd}")
  result = subprocess.run(cmd, shell=True, cwd=OPENPILOT_ROOT)
  if check and result.returncode != 0:
    print(f"Command failed with exit code {result.returncode}")
    sys.exit(result.returncode)
  return result


def main():
  parser = argparse.ArgumentParser(description="Prepare reference logs for ci-artifacts upload")
  parser.add_argument("--fork-username", type=str, required=True,
                      help="Your GitHub username (for ci-artifacts fork)")
  parser.add_argument("--skip-generate", action="store_true",
                      help="Skip generation, only prepare existing logs")
  args = parser.parse_args()

  print(f"OpenPilot root: {OPENPILOT_ROOT}")
  print(f"Output directory: {OUTPUT_DIR}")

  # Clean output directory
  if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
  OUTPUT_DIR.mkdir(parents=True)

  if not args.skip_generate:
    print("\n=== Generating reference logs ===")
    # Run the process replay with update-refs
    run_command(f"""
      uv run python selfdrive/test/process_replay/test_processes.py \\
        --update-refs \\
        -j$(nproc)
    """)

  # Copy generated files to output directory
  print("\n=== Preparing files for upload ===")
  if FAKEDATA_DIR.exists():
    zst_files = list(FAKEDATA_DIR.glob("*.zst"))
    print(f"Found {len(zst_files)} .zst files")
    
    for f in zst_files:
      shutil.copy(f, OUTPUT_DIR / f.name)
    
    ref_commit_file = PROC_REPLAY_DIR / "ref_commit"
    if ref_commit_file.exists():
      shutil.copy(ref_commit_file, OUTPUT_DIR / "ref_commit")
      with open(ref_commit_file) as f:
        commit = f.read().strip()
      print(f"Reference commit: {commit}")
  else:
    print("ERROR: fakedata directory not found. Run with --skip-generate=False first.")
    sys.exit(1)

  # Print upload instructions
  print("\n" + "="*60)
  print("UPLOAD INSTRUCTIONS")
  print("="*60)
  print(f"""
1. Clone your ci-artifacts fork:
   git clone https://github.com/{args.fork_username}/ci-artifacts.git
   cd ci-artifacts

2. Create a new branch:
   git checkout -b process-replay-update

3. Copy the generated files:
   cp {OUTPUT_DIR}/*.zst .
   cp {OUTPUT_DIR}/ref_commit .

4. Commit and push:
   git add *.zst ref_commit
   git commit -m "Update process replay refs for openpilot@$(cat {PROC_REPLAY_DIR}/ref_commit)"
   git push origin process-replay-update

5. Create a PR:
   Go to: https://github.com/commaai/ci-artifacts/compare/process-replay...{args.fork_username}:ci-artifacts:process-replay-update

6. After PR merges (or immediately for testing), update BASE_URL in test_processes.py:
   BASE_URL = "https://raw.githubusercontent.com/{args.fork_username}/ci-artifacts/refs/heads/process-replay/"
""")

  print(f"\nFiles ready for upload in: {OUTPUT_DIR}")
  print(f"Total files: {len(list(OUTPUT_DIR.iterdir()))}")


if __name__ == "__main__":
  main()
