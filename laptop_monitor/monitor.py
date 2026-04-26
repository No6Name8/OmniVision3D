"""
monitor.py — Laptop monitoring UI for OmniVision3D.

Receives drone-detected alerts from the ground Pi, prompts the operator
for a launch confirmation, then sends a launch command to the interceptor Pi.

Usage:
    python monitor.py --drone 192.168.1.200
    python monitor.py --drone 192.168.1.200 --port 5555
    python monitor.py --sim
"""

import argparse
import datetime
import logging
import select
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from alert_receiver  import AlertReceiver
from launch_command  import LaunchCommander

_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "monitor.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
_log = logging.getLogger("monitor")


# ---------------------------------------------------------------------------
def _banner(port: int, drone_ip: str, sim: bool) -> None:
    print("═" * 36)
    print("  LAPTOP MONITOR")
    print("═" * 36)
    print(f"  Listening on port: {port}")
    print(f"  Drone Pi IP:       {drone_ip}")
    print(f"  Press ENTER to confirm launch")
    print(f"  Press A      to abort")
    print(f"  Press Q      to quit")
    if sim:
        print(f"  Mode:            SIMULATION")
    print("═" * 36 + "\n")


def _ts(unix: float) -> str:
    return datetime.datetime.fromtimestamp(unix).strftime("%H:%M:%S")


def _stdin_ready() -> bool:
    """Non-blocking check for stdin input (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.kbhit()
    else:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(r)


def _read_char() -> str:
    """Read one character from stdin without waiting for Enter (Windows/Unix)."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        return ch if isinstance(ch, str) else ch.decode(errors="replace")
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    cfg     = yaml.safe_load(open(_ROOT / "config.yaml"))
    port    = args.port if args.port else cfg["comms"]["listen_port"]
    sim     = args.sim or cfg.get("simulation_mode", False)

    drone_ip = "127.0.0.1" if sim else (args.drone or cfg["comms"]["drone_ip"])

    _banner(port, drone_ip, sim)

    receiver  = AlertReceiver(port=port).start()
    commander = LaunchCommander(drone_ip, port=cfg["comms"]["drone_port"])

    pending_alert: dict | None = None   # alert awaiting operator ENTER
    launch_pending = False

    print("Monitoring active. Waiting for ground unit alerts...\n")

    try:
        while True:
            # ---- Check for new alert packet ----
            pkt = receiver.get_latest()

            if pkt and pkt.get("type") == "DRONE_DETECTED":
                conf     = pkt.get("confidence", 0.0)
                heading  = pkt.get("compass_heading")
                ts       = pkt.get("timestamp", time.time())
                source   = pkt.get("_from", pkt.get("source", "?"))

                print(f"\n{'!'*36}")
                print(f"!! DRONE DETECTED !!")
                print(f"  Confidence:  {conf:.0%}")
                print(f"  Direction:   {heading}°" if heading is not None else "  Direction:   unknown")
                print(f"  Time:        {_ts(ts)}")
                print(f"  Source:      {source}")
                print(f"{'!'*36}")
                print("\nPress ENTER to launch interceptor  |  A = abort  |  Q = quit")

                _log.info("DRONE_DETECTED from %s conf=%.3f heading=%s", source, conf, heading)
                pending_alert  = pkt
                launch_pending = True

            elif pkt and pkt.get("type") == "CLEAR":
                print(f"\n[{_ts(time.time())}] TARGET LOST — standing down\n")
                _log.info("CLEAR received")
                pending_alert  = None
                launch_pending = False

            # ---- Keyboard input ----
            if _stdin_ready():
                ch = _read_char()

                if ch in ("\r", "\n", ""):        # ENTER
                    if launch_pending and pending_alert:
                        commander.send_launch(pending_alert)
                        _log.info("LAUNCH confirmed by operator")
                        print("\n>> LAUNCH COMMAND SENT <<\n")
                        pending_alert  = None
                        launch_pending = False
                    else:
                        print()  # just newline if nothing pending

                elif ch.lower() == "a":
                    commander.send_abort()
                    _log.info("ABORT issued by operator")
                    pending_alert  = None
                    launch_pending = False

                elif ch.lower() == "q":
                    print("\nQuitting monitor.")
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        receiver.stop()
        commander.close()
        print("Monitor stopped.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D Laptop Monitor")
    parser.add_argument("--drone", type=str, default=None,
                        help="Interceptor Pi IP address")
    parser.add_argument("--port",  type=int, default=None,
                        help="UDP listen port (default 5555)")
    parser.add_argument("--sim",   action="store_true",
                        help="Simulation mode — launch commands go to localhost")
    args = parser.parse_args()
    run(args)
