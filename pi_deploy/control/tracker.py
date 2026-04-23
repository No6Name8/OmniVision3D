"""
tracker.py — State-machine target tracker for OmniVision3D.

States:
    LOCKED      : target visible, PID corrections active
    SEARCHING   : target lost < 3s, hold last correction
    NAVIGATING  : target lost 3–10s, fly to last known position
    ABORT       : target lost > 10s, return to base

TODO (MAVLink wiring):
    Replace _send_correction() stub with RC_CHANNELS_OVERRIDE commands.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(_LOG_DIR / "tracking.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)


class TrackState(str, Enum):
    LOCKED     = "LOCKED"
    SEARCHING  = "SEARCHING"
    NAVIGATING = "NAVIGATING"
    ABORT      = "ABORT"


@dataclass
class TrackingCommand:
    state:              TrackState
    pitch:              float = 0.0
    yaw:                float = 0.0
    dx:                 int   = 0
    dy:                 int   = 0
    lost_seconds:       float = 0.0
    last_known_center:  Optional[Tuple[int, int]] = None


class _PID:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral   = 0.0
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
        out              = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        self._prev_time  = now
        return float(out)


class Tracker:
    """
    Converts PipelineResult into pitch/yaw corrections with loss-state handling.
    """

    def __init__(self, config: dict) -> None:
        tcfg = config["tracking"]
        ccfg = config["camera"]

        self._frame_w  = ccfg["width"]
        self._frame_h  = ccfg["height"]
        self._t_search = tcfg["lost_searching_seconds"]
        self._t_nav    = tcfg["lost_navigating_seconds"]

        kp = tcfg["pid_p"]
        ki = tcfg["pid_i"]
        kd = tcfg["pid_d"]
        self._pid_yaw   = _PID(kp, ki, kd)
        self._pid_pitch = _PID(kp, ki, kd)

        self._last_pitch: float = 0.0
        self._last_yaw:   float = 0.0
        self._last_center: Optional[Tuple[int, int]] = None
        self._lost_since:  Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, result) -> TrackingCommand:
        """
        Compute next tracking command from a PipelineResult.

        Args:
            result: PipelineResult from VisionPipeline.process_frame()

        Returns:
            TrackingCommand with state, pitch, yaw, dx, dy.
        """
        from vision.pipeline import Phase

        if result.phase == Phase.LOCKED and result.detection is not None:
            self._lost_since = None
            self._pid_yaw.reset() if self._lost_since is not None else None

            cx, cy = result.detection.center
            frame_cx = self._frame_w / 2.0
            frame_cy = self._frame_h / 2.0

            dx = int(cx - frame_cx)
            dy = int(cy - frame_cy)

            norm_dx = dx / (self._frame_w  / 2.0)
            norm_dy = dy / (self._frame_h  / 2.0)

            yaw   = float(self._pid_yaw.update(norm_dx))
            pitch = float(self._pid_pitch.update(-norm_dy))

            yaw   = max(-1.0, min(1.0, yaw))
            pitch = max(-1.0, min(1.0, pitch))

            self._last_pitch  = pitch
            self._last_yaw    = yaw
            self._last_center = result.detection.center

            cmd = TrackingCommand(
                state=TrackState.LOCKED,
                pitch=pitch, yaw=yaw,
                dx=dx, dy=dy,
                last_known_center=self._last_center,
            )
            logging.info("LOCKED  pitch=%+.3f yaw=%+.3f dx=%d dy=%d", pitch, yaw, dx, dy)
            self._send_correction(pitch, yaw)
            return cmd

        # Target not LOCKED — start or continue lost timer
        if self._lost_since is None:
            self._lost_since = time.monotonic()
            self._pid_yaw.reset()
            self._pid_pitch.reset()

        lost = time.monotonic() - self._lost_since

        if lost < self._t_search:
            logging.info("SEARCHING  lost=%.1fs", lost)
            return TrackingCommand(
                state=TrackState.SEARCHING,
                pitch=self._last_pitch, yaw=self._last_yaw,
                lost_seconds=lost,
                last_known_center=self._last_center,
            )

        if lost < self._t_nav:
            logging.info("NAVIGATING  lost=%.1fs", lost)
            return TrackingCommand(
                state=TrackState.NAVIGATING,
                lost_seconds=lost,
                last_known_center=self._last_center,
            )

        logging.info("ABORT  lost=%.1fs", lost)
        return TrackingCommand(
            state=TrackState.ABORT,
            lost_seconds=lost,
            last_known_center=self._last_center,
        )

    def reset(self) -> None:
        self._lost_since   = None
        self._last_pitch   = 0.0
        self._last_yaw     = 0.0
        self._pid_yaw.reset()
        self._pid_pitch.reset()

    @staticmethod
    def _send_correction(pitch: float, yaw: float) -> None:
        # TODO: MAVLink RC_CHANNELS_OVERRIDE
        # master.mav.rc_channels_override_send(
        #     target_system, target_component,
        #     _yaw_to_rc(yaw), _pitch_to_rc(pitch),
        #     0, 0, 0, 0, 0, 0)
        pass
