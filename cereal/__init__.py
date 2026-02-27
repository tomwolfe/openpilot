import os
import capnp
import sys
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()

# Pre-load c++.capnp to avoid duplicate ID errors when loading other schemas
# that import it. This is a workaround for pycapnp not properly deduplicating
# imports when multiple schemas import the same file.
_ = capnp.load(os.path.join(CEREAL_PATH, "include/c++.capnp"))

# Load car schema from cereal directory first (before importing opendbc)
# This ensures that the path used for car.capnp is consistent
car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))

# Now load log and custom schemas
log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))
