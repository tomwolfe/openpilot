import os
import capnp
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()
  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))

# car schemas are sourced from opendbc package
# try to import from installed opendbc first, then fall back to local path
try:
  from opendbc.car.structs import car as opendbc_car
  car = opendbc_car
except ImportError:
  car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
