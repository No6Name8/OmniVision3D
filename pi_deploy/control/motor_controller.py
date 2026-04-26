"""
motor_controller.py — Power-differential motor controller for fixed-wing interceptor.

Motor layout (top-down view, nose forward):
    M1 (front-left)   M2 (front-right)
    M3 (rear-left)    M4 (rear-right)

Turn logic (positive correction = RIGHT turn / nose right):
    Right turn: M1 and M3 increase, M2 and M4 decrease
    Left turn:  M1 and M3 decrease, M2 and M4 increase

    M1 = base + yaw_us + pitch_us
    M2 = base - yaw_us + pitch_us
    M3 = base + yaw_us - pitch_us
    M4 = base - yaw_us - pitch_us

Pitch logic (positive correction = pitch UP / nose up):
    Pitch up:   M1 and M2 increase (front), M3 and M4 decrease (rear)
    Pitch down: M1 and M2 decrease, M3 and M4 increase

PID error convention:
    yaw error   = dx           (positive = target right  → need right turn)
    pitch error = -dy          (positive = target above  → need pitch up)

All output values are in PWM microseconds (1000–2000 us, neutral = 1500 us).
Motor wiring and PID gains will be validated and tuned during real flight tests.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fh = logging.FileHandler(str(_LOG_DIR / "motor_commands.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_log = logging.getLogger("motor_controller")
_log.setLevel(logging.INFO)
_log.addHandler(_fh)
_log.propagate = False

# CSV log
_csv_path = _LOG_DIR / "motor_commands.csv"
_csv_f    = open(_csv_path, "a", newline="")
_csv_w    = csv.writer(_csv_f)


class MotorController:
    """
    Converts pixel offsets (dx, dy) into per-channel PWM microsecond commands.
    Runs a discrete PID loop independently for yaw and pitch.
    """

    def __init__(self, config: dict) -> None:
        mcfg = config.get("motor_control", {})
        mmcfg = config.get("motor_mapping", {})

        self._base_us         = int(mcfg.get("base_power_us",    1500))
        self._min_us          = int(mcfg.get("min_power_us",     1000))
        self._max_us          = int(mcfg.get("max_power_us",     2000))
        self._max_correction  = float(mcfg.get("max_correction_pct", 20.0))
        self._max_px          = float(mcfg.get("max_pixel_offset",   320.0))
        self._sim             = mcfg.get("simulation_mode", True)

        pid = mcfg.get("pid", {})
        self._kp = float(pid.get("Kp", 0.06))
        self._ki = float(pid.get("Ki", 0.001))
        self._kd = float(pid.get("Kd", 0.01))

        # PID state — yaw and pitch are independent
        self._yaw_integral:   float = 0.0
        self._yaw_last_error: float = 0.0
        self._pitch_integral:   float = 0.0
        self._pitch_last_error: float = 0.0

        # Motor → channel mapping (which physical channel is each motor)
        # Values are 1-indexed channel numbers
        self._mapping = {
            "front_left":  self._parse_ch(mmcfg.get("front_left",  "channel_1")),
            "front_right": self._parse_ch(mmcfg.get("front_right", "channel_2")),
            "rear_left":   self._parse_ch(mmcfg.get("rear_left",   "channel_3")),
            "rear_right":  self._parse_ch(mmcfg.get("rear_right",  "channel_4")),
        }
        self._flip_yaw   = bool(mmcfg.get("flip_yaw",   False))
        self._flip_pitch = bool(mmcfg.get("flip_pitch", False))

        print(f"[MotorController] Base: {self._base_us}us  "
              f"Max correction: {self._max_correction:.0f}%  "
              f"PID Kp={self._kp} Ki={self._ki} Kd={self._kd}")
        print(f"[MotorController] Mapping: "
              f"FL=CH{self._mapping['front_left']} "
              f"FR=CH{self._mapping['front_right']} "
              f"RL=CH{self._mapping['rear_left']} "
              f"RR=CH{self._mapping['rear_right']}")
        if self._sim:
            print("[MotorController] SIMULATION MODE (no hardware commands)")

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ch(value) -> int:
        """Accept 'channel_2' or bare int 2."""
        if isinstance(value, int):
            return value
        return int(str(value).replace("channel_", ""))

    # ------------------------------------------------------------------
    def calculate_corrections(self, dx: int, dy: int) -> Dict[str, float]:
        """
        Run PID for yaw and pitch and return correction percentages.

        dx > 0 → target is right of centre → positive yaw correction (right turn)
        dy > 0 → target is below centre    → negative pitch correction (pitch down)
        dy < 0 → target is above centre    → positive pitch correction (pitch up)
        """
        # Yaw PID (error = dx, positive → turn right)
        yaw_p = self._kp * dx
        self._yaw_integral  += self._ki * dx
        yaw_d = self._kd * (dx - self._yaw_last_error)
        yaw_correction = yaw_p + self._yaw_integral + yaw_d
        self._yaw_last_error = dx

        # Pitch PID (error = -dy so positive = pitch up)
        pitch_err = -dy
        pitch_p = self._kp * pitch_err
        self._pitch_integral  += self._ki * pitch_err
        pitch_d = self._kd * (pitch_err - self._pitch_last_error)
        pitch_correction = pitch_p + self._pitch_integral + pitch_d
        self._pitch_last_error = pitch_err

        # Apply flip flags (set by calibration if wiring is reversed)
        if self._flip_yaw:
            yaw_correction = -yaw_correction
        if self._flip_pitch:
            pitch_correction = -pitch_correction

        # Clamp to ±max_correction %
        yaw_correction   = max(-self._max_correction, min(self._max_correction, yaw_correction))
        pitch_correction = max(-self._max_correction, min(self._max_correction, pitch_correction))

        return {"yaw": yaw_correction, "pitch": pitch_correction}

    # ------------------------------------------------------------------
    def calculate_motor_powers(self, dx: int, dy: int) -> Dict:
        """
        Convert pixel offset into per-channel PWM microsecond values.

        Motor power formulas (right turn = positive yaw_us):
            M1 front-left  = base + yaw_us + pitch_us
            M2 front-right = base - yaw_us + pitch_us
            M3 rear-left   = base + yaw_us - pitch_us
            M4 rear-right  = base - yaw_us - pitch_us
        """
        corrections = self.calculate_corrections(dx, dy)
        yaw_pct     = corrections["yaw"]
        pitch_pct   = corrections["pitch"]

        # Convert percentage to microseconds (relative to base)
        yaw_us   = (yaw_pct   / 100.0) * self._base_us
        pitch_us = (pitch_pct / 100.0) * self._base_us

        # Logical motor values
        m1 = self._base_us + yaw_us + pitch_us   # front-left
        m2 = self._base_us - yaw_us + pitch_us   # front-right
        m3 = self._base_us + yaw_us - pitch_us   # rear-left
        m4 = self._base_us - yaw_us - pitch_us   # rear-right

        motors = {
            "front_left":  m1,
            "front_right": m2,
            "rear_left":   m3,
            "rear_right":  m4,
        }

        # Map logical motors → physical channels
        channels = {}
        for motor_name, ch_num in self._mapping.items():
            us_val = int(max(self._min_us, min(self._max_us, motors[motor_name])))
            channels[f"channel_{ch_num}"] = us_val

        # Ensure all 4 channels are present (fill any unmapped with base)
        for i in range(1, 5):
            channels.setdefault(f"channel_{i}", self._base_us)

        return {
            "channel_1":           channels["channel_1"],
            "channel_2":           channels["channel_2"],
            "channel_3":           channels["channel_3"],
            "channel_4":           channels["channel_4"],
            "yaw_correction_pct":  yaw_pct,
            "pitch_correction_pct": pitch_pct,
            "dx_pixels":           dx,
            "dy_pixels":           dy,
        }

    # ------------------------------------------------------------------
    def send_to_pixhawk(self, motor_powers: Dict) -> None:
        """
        SIMULATION: print formatted motor commands and log to CSV.
        TODO: replace with MAVLink RC_CHANNELS_OVERRIDE when Pixhawk is wired.
        """
        ch1 = motor_powers["channel_1"]
        ch2 = motor_powers["channel_2"]
        ch3 = motor_powers["channel_3"]
        ch4 = motor_powers["channel_4"]
        yaw_pct   = motor_powers["yaw_correction_pct"]
        pitch_pct = motor_powers["pitch_correction_pct"]

        if self._sim:
            print(f"  MOTOR COMMANDS:")
            print(f"    CH1: {ch1}us  ({yaw_pct:+.1f}% yaw)")
            print(f"    CH2: {ch2}us")
            print(f"    CH3: {ch3}us")
            print(f"    CH4: {ch4}us  ({pitch_pct:+.1f}% pitch)")

        _log.info("CH1=%d CH2=%d CH3=%d CH4=%d yaw=%+.2f%% pitch=%+.2f%%",
                  ch1, ch2, ch3, ch4, yaw_pct, pitch_pct)

        _csv_w.writerow([
            f"{time.time():.3f}",
            ch1, ch2, ch3, ch4,
            f"{yaw_pct:.3f}", f"{pitch_pct:.3f}",
            motor_powers["dx_pixels"], motor_powers["dy_pixels"],
        ])
        _csv_f.flush()

        # TODO: MAVLink wiring
        # master.mav.rc_channels_override_send(
        #     target_system, target_component,
        #     ch1, ch2, ch3, ch4, 0, 0, 0, 0)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all PID state. Call when target is lost or mission ends."""
        self._yaw_integral    = 0.0
        self._yaw_last_error  = 0.0
        self._pitch_integral  = 0.0
        self._pitch_last_error = 0.0
