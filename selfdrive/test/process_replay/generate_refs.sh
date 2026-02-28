#!/bin/bash
# Generate process replay reference logs
# Run this on a Linux machine with openpilot dependencies installed

set -e

echo "=== Generating Process Replay Reference Logs ==="
echo ""

# Navigate to openpilot directory
cd "$(dirname "$0")/.."

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
  echo "ERROR: This script must be run on Linux"
  echo "The process replay tests require msgq which only works on Linux"
  echo ""
  echo "Options:"
  echo "1. Run this script on a Linux machine"
  echo "2. Run in GitHub Actions (see .github/workflows/update_process_replay_refs.yml)"
  echo "3. Run in Docker: docker run -v \$(pwd):/openpilot commaai/openpilot-base ./selfdrive/test/process_replay/generate_refs.sh"
  exit 1
fi

# Check dependencies
if ! command -v scons &> /dev/null; then
  echo "Installing dependencies..."
  ./tools/op.sh setup
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
