"""
tracker.py — PID-based target tracker for OmniVision3D.

Takes a pixel offset (dx, dy) from the frame centre and returns normalised
pitch and yaw corrections in [-1.0, 1.0].

Simulation mode: corrections are printed and logged; no hardware is touched.

TODO (when flight controller is wired):
    - Replace _send_correction() stub with MAVLink RC-override calls.
    - Tune PID gains against real airframe response.
    - Add roll axis if the gimbal supports it.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(_LOG_DIR / "tracking.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


class PIDController:
    """Single-axis discrete PID controller."""

    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral  = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.monotonic()

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.monotonic()

    def update(self, error: float) -> float:
        now = time.monotonic()
        dt  = max(now - self._prev_time, 1e-4)

        self._integral  += error * dt
        derivative       = (error - self._prev_error) / dt
        output           = self.kp * error + self.ki * self._integral + self.kd * derivative

        self._prev_error = error
        self._prev_time  = now
        return float(output)


class TargetTracker:
    """
    Converts frame-centre pixel offsets into pitch/yaw corrections.

    Corrections are clamped to [-1.0, 1.0] where ±1.0 represents the
    maximum command sent to the flight controller.

    If the target is lost, the last valid correction is held for
    `hold_duration` seconds before returning to (0.0, 0.0) scan state.
    """

    def __init__(
        self,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        hold_duration: float = 1.0,
        simulation: bool = True,
    ) -> None:
        self._pitch_pid = PIDController(kp, ki, kd)
        self._yaw_pid   = PIDController(kp, ki, kd)
        self._hold      = hold_duration
        self._sim       = simulation

        self._last_correction: Tuple[float, float] = (0.0, 0.0)
        self._lost_since: float = 0.0
        self._tracking  = False

    def update(
        self,
        offset: Tuple[float, float],
        target_visible: bool,
    ) -> Tuple[float, float]:
        """
        Compute pitch and yaw corrections.

        Args:
            offset:         (dx, dy) pixels from frame centre.  dx>0 = target right.
            target_visible: True when a drone detection exceeds the confidence threshold.

        Returns:
            (pitch, yaw) corrections in [-1.0, 1.0].
            Positive pitch  = nose up.
            Positive yaw    = nose right.
        """
        if target_visible:
            dx, dy     = offset
            yaw_out    = float(self._yaw_pid.update(dx))
            pitch_out  = float(self._pitch_pid.update(-dy))  # dy>0 = target below → pitch down

            yaw_out   = max(-1.0, min(1.0, yaw_out))
            pitch_out = max(-1.0, min(1.0, pitch_out))

            self._last_correction = (pitch_out, yaw_out)
            self._lost_since      = 0.0
            self._tracking        = True

            self._send_correction(pitch_out, yaw_out, "TRACKING")
            return pitch_out, yaw_out

        # Target lost
        if self._tracking and self._lost_since == 0.0:
            self._lost_since = time.monotonic()

        held_for = time.monotonic() - self._lost_since if self._lost_since else 0.0

        if held_for < self._hold:
            pitch_out, yaw_out = self._last_correction
            self._send_correction(pitch_out, yaw_out, f"HOLD ({held_for:.1f}s)")
            return pitch_out, yaw_out

        # Hold expired — return to scan
        if self._tracking:
            logging.info("Target lost — returning to SCAN")
            self._tracking = False
            self._pitch_pid.reset()
            self._yaw_pid.reset()
            self._last_correction = (0.0, 0.0)
            self._lost_since      = 0.0

        self._send_correction(0.0, 0.0, "SCANNING")
        return 0.0, 0.0

    def _send_correction(self, pitch: float, yaw: float, state: str) -> None:
        msg = f"{state:<12}  pitch={pitch:+.3f}  yaw={yaw:+.3f}"
        logging.info(msg)
        if self._sim:
            pass  # mission loop prints; avoid duplicate stdout noise
        # TODO: replace with MAVLink RC-override when flight controller is wired
        # Example:
        #   master.mav.rc_channels_override_send(
        #       master.target_system, master.target_component,
        #       _yaw_to_rc(yaw), _pitch_to_rc(pitch), 0, 0, 0, 0, 0, 0)
