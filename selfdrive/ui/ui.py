#!/usr/bin/env python3
import os

from openpilot.system.hardware import HARDWARE, TICI
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  # Use capability-based core affinity check
  cores = {5, }
  config_realtime_process(0, 51)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  for should_render in gui_app.render():
    ui_state.update()
    if should_render:
      # Re-affine after power save offlines our core
      # Only on hardware with core affinity control
      if HARDWARE.capabilities.has_core_affinity_control:
        try:
          current_cores = os.sched_getaffinity(0)
          if current_cores != cores:
            set_core_affinity(list(cores))
        except OSError:
          pass


if __name__ == "__main__":
  main()
