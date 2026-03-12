#!/usr/bin/env python3
from enum import IntEnum
from typing import Optional


# TODO: this should be automatically determined using the capnp schema
class QueueSize(IntEnum):
  BIG = 10 * 1024 * 1024      # 10MB - video frames, large AI outputs
  MEDIUM = 2 * 1024 * 1024    # 2MB - high freq (CAN), livestream
  SMALL = 250 * 1024          # 250KB - most services


class Service:
  def __init__(self, should_log: bool, frequency: float, decimation: Optional[int] = None,
               queue_size: QueueSize = QueueSize.SMALL):
    self.should_log = should_log
    self.frequency = frequency
    self.decimation = decimation
    self.queue_size = queue_size


_services: dict[str, tuple] = {
  # service: (should_log, frequency, qlog decimation (optional))
  # note: the "EncodeIdx" packets will still be in the log
  # E2E Phase 1: Optimized decimation for 100KB qlog target
  "gyroscope": (True, 104., 208),  # Increased decimation for E2E telemetry optimization
  "accelerometer": (True, 104., 208),  # Increased decimation for E2E telemetry optimization
  "magnetometer": (True, 25., 50),  # Increased decimation
  "lightSensor": (True, 100., 200),  # Increased decimation
  "temperatureSensor": (True, 2., 400),  # Increased decimation
  "gpsNMEA": (True, 9., 18),  # GPS decimated - not used for driving logic
  "deviceState": (True, 2., 2),  # Slightly increased decimation
  "touch": (True, 20., 20),  # Increased decimation
  "can": (True, 100., 4106, QueueSize.BIG),  # Increased decimation for ~1.5 msgs in full segment
  "controlsState": (True, 100., 20, QueueSize.MEDIUM),  # Increased decimation
  "selfdriveState": (True, 100., 20),  # Increased decimation
  "pandaStates": (True, 10., 2),  # Increased decimation
  "peripheralState": (True, 2., 2),  # Increased decimation
  "radarState": (True, 20., 10),  # Increased decimation
  "roadEncodeIdx": (False, 20., 1),
  "liveTracks": (True, 20., 40),  # Increased decimation
  "sendcan": (True, 100., 278, QueueSize.MEDIUM),  # Increased decimation
  "logMessage": (True, 0.),
  "errorLogMessage": (True, 0., 1),
  "liveCalibration": (True, 4., 8),  # Increased decimation
  "liveTorqueParameters": (True, 4., 2),  # Increased decimation
  "liveDelay": (True, 4., 2),  # Increased decimation
  "androidLog": (True, 0.),
  "carState": (True, 100., 20),  # Increased decimation
  "carControl": (True, 100., 20),  # Increased decimation
  "carOutput": (True, 100., 20),  # Increased decimation
  "longitudinalPlan": (True, 20., 20),  # Increased decimation
  "driverAssistance": (True, 20., 40),  # Increased decimation
  "procLog": (True, 0.5, 30, QueueSize.BIG),  # Increased decimation
  "gpsLocationExternal": (True, 10., 20),  # GPS decimated - not used for driving logic
  "gpsLocation": (True, 1., 2),  # GPS decimated - not used for driving logic
  "ubloxGnss": (True, 10., 20),  # GPS decimated - not used for driving logic
  "qcomGnss": (True, 2., 4),  # GPS decimated - not used for driving logic
  "gnssMeasurements": (True, 10., 20),  # GPS decimated - not used for driving logic
  "clocks": (True, 0.1, 2),  # Increased decimation
  "ubloxRaw": (True, 20., 40),  # GPS decimated - not used for driving logic
  "livePose": (True, 20., 8),  # Increased decimation - primary driving input
  "liveParameters": (True, 20., 10),  # Increased decimation
  "cameraOdometry": (True, 20., 20),  # Increased decimation - primary driving input
  "thumbnail": (True, 1 / 60., 1),
  "onroadEvents": (True, 1., 2),  # Increased decimation
  "carParams": (True, 0.02, 2),  # Increased decimation
  "roadCameraState": (True, 20., 40),  # Increased decimation
  "driverCameraState": (True, 20., 40),  # Increased decimation
  "driverEncodeIdx": (False, 20., 1),
  "driverStateV2": (True, 20., 20),  # Increased decimation
  "driverMonitoringState": (True, 20., 20),  # Increased decimation
  "wideRoadEncodeIdx": (False, 20., 1),
  "wideRoadCameraState": (True, 20., 40),  # Increased decimation
  "drivingModelData": (True, 20., 20),  # Increased decimation - critical for E2E
  "modelV2": (True, 20., None, QueueSize.BIG),  # Keep full resolution for E2E model output
  "navEmbeddings": (True, 20., 40),  # E2E Phase 4: Navigation embeddings for semantic routing
  "managerState": (True, 2., 2),  # Increased decimation
  "uploaderState": (True, 0., 1),
  "navInstruction": (True, 1., 20),  # Increased decimation
  "navRoute": (True, 0.),
  "navThumbnail": (True, 0.),
  "qRoadEncodeIdx": (False, 20.),
  "userBookmark": (True, 0., 1),
  "soundPressure": (True, 10., 20),  # Increased decimation
  "rawAudioData": (False, 20.),  # Not in qlog - high bandwidth
  "bookmarkButton": (True, 0., 1),
  "audioFeedback": (True, 0., 1),
  "roadEncodeData": (False, 20., None, QueueSize.BIG),
  "driverEncodeData": (False, 20., None, QueueSize.BIG),
  "wideRoadEncodeData": (False, 20., None, QueueSize.BIG),
  "qRoadEncodeData": (False, 20., None, QueueSize.BIG),

  # debug
  "uiDebug": (True, 0., 1),
  "testJoystick": (True, 0.),
  "alertDebug": (True, 20., 10),  # Increased decimation
  "livestreamWideRoadEncodeIdx": (False, 20.),
  "livestreamRoadEncodeIdx": (False, 20.),
  "livestreamDriverEncodeIdx": (False, 20.),
  "livestreamWideRoadEncodeData": (False, 20., None, QueueSize.MEDIUM),
  "livestreamRoadEncodeData": (False, 20., None, QueueSize.MEDIUM),
  "livestreamDriverEncodeData": (False, 20., None, QueueSize.MEDIUM),
  "customReservedRawData0": (True, 0.),
  "customReservedRawData1": (True, 0.),
  "customReservedRawData2": (True, 0.),
}
SERVICE_LIST = {name: Service(*vals) for
                idx, (name, vals) in enumerate(_services.items())}


def build_header():
  h = ""
  h += "/* THIS IS AN AUTOGENERATED FILE, PLEASE EDIT services.py */\n"
  h += "#ifndef __SERVICES_H\n"
  h += "#define __SERVICES_H\n"

  h += "#include <map>\n"
  h += "#include <string>\n"

  h += "struct service { std::string name; bool should_log; float frequency; int decimation; size_t queue_size; };\n"
  h += "static std::map<std::string, service> services = {\n"
  for k, v in SERVICE_LIST.items():
    should_log = "true" if v.should_log else "false"
    decimation = -1 if v.decimation is None else v.decimation
    h += '  { "%s", {"%s", %s, %f, %d, %d}},\n' % \
         (k, k, should_log, v.frequency, decimation, v.queue_size)
  h += "};\n"

  h += "#endif\n"
  return h


if __name__ == "__main__":
  print(build_header())
