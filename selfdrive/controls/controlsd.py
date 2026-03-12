#!/usr/bin/env python3
import math
from numbers import Number

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_e2e import LatControlE2E
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longcontrol_e2e import LongControlE2E
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())


class Controls:
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    self.CI = interfaces[self.CP.carFingerprint](self.CP)

    self.sm = messaging.SubMaster(['liveDelay', 'liveParameters', 'liveTorqueParameters', 'modelV2', 'selfdriveState',
                                   'liveCalibration', 'livePose', 'longitudinalPlan', 'carState', 'carOutput',
                                   'driverMonitoringState', 'onroadEvents', 'driverAssistance'], poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'])

    self.steer_limited_by_safety = False
    self.curvature = 0.0
    self.desired_curvature = 0.0

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    # E2E Phase 2: Check if E2E direct actuation mode is enabled
    self.e2e_enabled = self.params.get_bool("E2E_Enabled")
    
    if self.e2e_enabled:
      cloudlog.info("E2E Phase 2: Direct actuation mode enabled")
      self.LoC = LongControlE2E(self.CP)
      self.LaC = LatControlE2E(self.CP, self.CI, DT_CTRL)
    else:
      cloudlog.info("Classical control mode (PID/MPC)")
      self.LoC = LongControl(self.CP)
      self.VM = VehicleModel(self.CP)
      if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.LaC = LatControlAngle(self.CP, self.CI, DT_CTRL)
      elif self.CP.lateralTuning.which() == 'pid':
        self.LaC = LatControlPID(self.CP, self.CI, DT_CTRL)
      elif self.CP.lateralTuning.which() == 'torque':
        self.LaC = LatControlTorque(self.CP, self.CI, DT_CTRL)

  def update(self):
    self.sm.update(15)
    if self.sm.updated["liveCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated["livePose"]:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

  def state_control(self):
    CS = self.sm['carState']

    # Update VehicleModel (only needed for classical control)
    if not self.e2e_enabled:
      lp = self.sm['liveParameters']
      x = max(lp.stiffnessFactor, 0.1)
      sr = max(lp.steerRatio, 0.1)
      self.VM.update_params(x, sr)

      steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
      self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)

      # Update Torque Params
      if self.CP.lateralTuning.which() == 'torque':
        torque_params = self.sm['liveTorqueParameters']
        if self.sm.all_checks(['liveTorqueParameters']) and torque_params.useParams:
          self.LaC.update_live_torque_params(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                             torque_params.frictionCoefficientFiltered)

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
    CC.latActive = self.sm['selfdriveState'].active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and self.CP.openpilotLongitudinalControl

    actuators = CC.actuators
    actuators.longControlState = self.LoC.long_control_state

    # Enable blinkers while lane changing
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # E2E Phase 2: Extract model actuator predictions
    model_actuator_output = {
      'steer_torque_pred': model_v2.action.steerTorquePred,
      'steer_angle_pred': model_v2.action.steerAnglePred,
      'gas_pred': model_v2.action.gasPred,
      'brake_pred': model_v2.action.brakePred,
      'crash_prob': model_v2.action.crashProb,
      'ttc_pred': model_v2.action.ttcPred,
    }
    
    # E2E AEB: Check for imminent collision and prepare override
    aeb_override = None
    if model_actuator_output['crash_prob'] > 0.7 or (model_actuator_output['ttc_pred'] > 0 and model_actuator_output['ttc_pred'] < 1.5):
      # Imminent collision detected - apply maximum braking
      aeb_override = -4.0  # Maximum braking acceleration (m/s^2)

    # accel control (E2E or classical)
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    
    if self.e2e_enabled:
      # E2E Phase 2: Direct actuation
      actuators.accel = float(self.LoC.update(CC.longActive, CS, model_actuator_output, pid_accel_limits, aeb_override))
    else:
      # Classical: accel PID loop
      actuators.accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits))

    # Steering control (E2E or classical)
    if self.e2e_enabled:
      # E2E Phase 2: Direct actuation
      lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
      steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM if hasattr(self, 'VM') else None,
                                                         self.sm['liveParameters'] if not self.e2e_enabled else None,
                                                         self.steer_limited_by_safety, 0.0, False, lat_delay,
                                                         model_actuator_output)
      actuators.torque = float(steer)
      actuators.steeringAngleDeg = float(steeringAngleDeg)
    else:
      # Classical: Steering PID loop and lateral MPC
      # Reset desired curvature to current to avoid violating the limits on engage
      new_desired_curvature = model_v2.action.desiredCurvature if CC.latActive else self.curvature
      self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, self.sm['liveParameters'].roll)
      lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS

      actuators.curvature = self.desired_curvature
      steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, self.sm['liveParameters'],
                                                         self.steer_limited_by_safety, self.desired_curvature,
                                                         curvature_limited, lat_delay)
      actuators.torque = float(steer)
      actuators.steeringAngleDeg = float(steeringAngleDeg)
    
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and self.CP.openpilotLongitudinalControl
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise)
    CC.cruiseControl.resume = CC.enabled and CS.cruiseState.standstill and not self.sm['longitudinalPlan'].shouldStop

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    if self.sm['selfdriveState'].active:
      CO = self.sm['carOutput']
      if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.steer_limited_by_safety = abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > \
                                              STEER_ANGLE_SATURATION_THRESHOLD
      else:
        self.steer_limited_by_safety = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    
    # E2E Phase 2: Log E2E-specific state
    if self.e2e_enabled:
      cs.e2EEnabled = True
      cs.aebActive = self.LoC.get_aeb_status() if hasattr(self.LoC, 'get_aeb_status') else False
      # E2E controllers don't use PID, so set to 0
      cs.upAccelCmd = 0.0
      cs.uiAccelCmd = 0.0
      cs.ufAccelCmd = 0.0
    else:
      cs.e2EEnabled = False
      cs.aebActive = False
      cs.upAccelCmd = float(self.LoC.pid.p)
      cs.uiAccelCmd = float(self.LoC.pid.i)
      cs.ufAccelCmd = float(self.LoC.pid.f)
    
    cs.forceDecel = bool((self.sm['driverMonitoringState'].awarenessStatus < 0.) or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    # Determine lateral tuning type for logging
    lat_tuning = self.CP.lateralTuning.which() if not self.e2e_enabled else 'e2e'
    
    if self.e2e_enabled:
      # E2E lateral control state
      cs.lateralControlState.e2EState = lac_log
    elif self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif lat_tuning == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_tuning == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      self.update()
      CC, lac_log = self.state_control()
      self.publish(CC, lac_log)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
