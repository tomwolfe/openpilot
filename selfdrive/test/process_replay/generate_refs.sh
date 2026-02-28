#!/bin/bash
# Generate process replay reference logs
# Run this on a Linux machine with openpilot dependencies installed

set -e

echo "=== Generating Process Replay Reference Logs ==="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to openpilot root (three levels up: process_replay -> test -> selfdrive -> root)
OPENPILOT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$OPENPILOT_ROOT"

echo "Working directory: $(pwd)"
echo "Script directory: $SCRIPT_DIR"

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
  echo "WARNING: Running on $(uname), not Linux"
  echo "The process replay tests may not work correctly on non-Linux systems"
  echo ""
  echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
  sleep 5
fi

# Check if op.sh exists
if [[ ! -f "tools/op.sh" ]]; then
  echo "ERROR: tools/op.sh not found"
  echo "Make sure you're running this from within the openpilot repository"
  exit 1
fi

# Check dependencies - skip op.sh setup in Codespaces
if [[ -n "$CODESPACES" ]]; then
  echo "Running in GitHub Codespaces, skipping op.sh setup"
  echo "Dependencies should already be installed"
else
  if ! command -v scons &> /dev/null; then
    echo "Installing dependencies..."
    bash tools/op.sh setup || {
      echo "WARNING: op.sh setup failed, continuing anyway..."
      echo "You may need to install dependencies manually"
    }
  fi
fi

# Build openpilot
echo "Building openpilot..."
scons -j$(nproc)

# Generate reference logs
echo ""
echo "Generating reference logs..."
uv run python selfdrive/test/process_replay/test_processes.py --update-refs -j$(nproc)

# Show results
echo ""
echo "=== Generation Complete ==="
echo ""
echo "Reference commit: $(cat selfdrive/test/process_replay/ref_commit)"
echo ""
echo "Generated files:"
ls -lh selfdrive/test/process_replay/fakedata/*.zst | wc -l
echo ""
echo "Next steps:"
echo "1. Copy these files to your ci-artifacts fork:"
echo "   - All .zst files from selfdrive/test/process_replay/fakedata/"
echo "   - selfdrive/test/process_replay/ref_commit"
echo ""
echo "2. Push to your fork's process-replay branch:"
echo "   cd /path/to/ci-artifacts-fork"
echo "   git checkout -b process-replay"
echo "   cp /path/to/openpilot/selfdrive/test/process_replay/fakedata/*.zst ."
echo "   cp /path/to/openpilot/selfdrive/test/process_replay/ref_commit ."
echo "   git add *.zst ref_commit"
echo "   git commit -m 'Update process replay refs for openpilot@$(cat selfdrive/test/process_replay/ref_commit)'"
echo "   git push origin process-replay"
echo ""
echo "3. Set GitHub variable in your openpilot fork:"
echo "   Settings -> Variables -> New variable"
echo "   Name: CI_ARTIFACTS_BASE_URL"
echo "   Value: https://raw.githubusercontent.com/YOUR_USERNAME/ci-artifacts/refs/heads/process-replay/"
