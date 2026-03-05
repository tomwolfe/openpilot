#!/bin/bash
# Regenerate remaining segments one at a time to avoid memory issues

set -e

FAKEDATA="selfdrive/test/process_replay/fakedata"

echo "=== Regenerating FORD segment ==="
container run --rm --volume /Users/tom/Documents/apps/openpilot:/openpilot openpilot-local \
  python selfdrive/test/process_replay/regen.py "regen755D8CB1E1F|2025-04-08--23-13-43--0" \
  --whitelist-procs controlsd,plannerd,card,radard,lagd

echo "=== Regenerating RIVIAN segment ==="
container run --rm --volume /Users/tom/Documents/apps/openpilot:/openpilot openpilot-local \
  python selfdrive/test/process_replay/regen.py "regen5FCAC896BBE|2025-04-08--23-13-35--0" \
  --whitelist-procs controlsd,plannerd,card,radard,lagd

echo "=== Regenerating TESLA segment ==="
container run --rm --volume /Users/tom/Documents/apps/openpilot:/openpilot openpilot-local \
  python selfdrive/test/process_replay/regen.py "2c912ca5de3b1ee9|0000025d--6eb6bcbca4--4" \
  --whitelist-procs controlsd,plannerd,card,radard,lagd

echo "=== All segments regenerated ==="
