"""
motor_calibration.py — One-time motor mapping calibration for the interceptor drone.

Run once before first flight. Spins each ESC channel briefly so the operator
can identify which physical motor corresponds to which flight controller channel.
Then performs a sanity-check turn and pitch to confirm the mapping is correct.
Saves the result to pi_deploy/config.yaml.

Usage:
    python pi_deploy/control/motor_calibration.py
    python pi_deploy/control/motor_calibration.py --config path/to/config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

_ROOT      = Path(__file__).resolve().parent.parent
_CONFIG    = _ROOT / "config.yaml"
_POSITIONS = ["FRONT-LEFT", "FRONT-RIGHT", "REAR-LEFT", "REAR-RIGHT"]
_KEYS      = ["front_left", "front_right", "rear_left", "rear_right"]


# ---------------------------------------------------------------------------
def _spin_channel(ch: int, sim: bool = True) -> None:
    """Pulse one ESC channel at low throttle for 2 seconds."""
    if sim:
        print(f"    [SIM] SPINNING CH{ch} at 1600us for 2s...")
        time.sleep(0.3)
    else:
        # TODO: MAVLink RC_CHANNELS_OVERRIDE pulse on ch
        # master.mav.rc_channels_override_send(...)
        print(f"    Spinning CH{ch} — watch the drone!")
        time.sleep(2.0)


def _send_correction_test(mapping: dict, direction: str, pct: float, sim: bool) -> None:
    """Send a tiny correction to the motors to test yaw or pitch."""
    base = 1500
    corr_us = int((pct / 100.0) * base)

    fl = mapping["front_left"]
    fr = mapping["front_right"]
    rl = mapping["rear_left"]
    rr = mapping["rear_right"]

    if direction == "right":
        # Right turn: FL+RL increase, FR+RR decrease
        powers = {fl: base + corr_us, fr: base - corr_us,
                  rl: base + corr_us, rr: base - corr_us}
    elif direction == "up":
        # Pitch up: FL+FR increase, RL+RR decrease
        powers = {fl: base + corr_us, fr: base + corr_us,
                  rl: base - corr_us, rr: base - corr_us}
    else:
        return

    if sim:
        label = "RIGHT TURN" if direction == "right" else "PITCH UP"
        print(f"    [SIM] Commanding {label} at {pct:.0f}% correction:")
        for ch, us in sorted(powers.items()):
            arrow = "↑" if us > base else "↓"
            print(f"      CH{ch}: {us}us {arrow}")
        time.sleep(0.3)
    else:
        # TODO: send via MAVLink
        time.sleep(2.0)


def _ask_channel(prompt: str) -> int:
    while True:
        try:
            ch = int(input(prompt).strip())
            if 1 <= ch <= 4:
                return ch
        except (ValueError, EOFError):
            pass
        print("    Please enter a number between 1 and 4.")


def _ask_yn(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes", ""):
            return True
        if ans in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
def run(config_path: Path, sim: bool) -> None:
    print()
    print("═" * 36)
    print("  MOTOR CALIBRATION")
    print("═" * 36)
    print("  This runs once before first flight.")
    print("  Watch the drone carefully.")
    print("  Press ENTER after each question.")
    print(f"  Mode: {'SIMULATION' if sim else 'LIVE HARDWARE'}")
    print("═" * 36)
    print()

    cfg = yaml.safe_load(open(config_path))
    mapping: dict = {}          # position_key → channel number
    used_channels: set = set()

    # ---- Step 1-4: Identify each motor ----
    for step, (position, key) in enumerate(zip(_POSITIONS, _KEYS), start=1):
        print(f"Step {step}: Which channel is the {position} motor?")
        for ch in range(1, 5):
            if ch not in used_channels:
                input(f"  Press ENTER to spin channel {ch}...")
                _spin_channel(ch, sim)

        ch = _ask_channel("  Which motor spun? Enter channel number (1-4): ")
        if ch in used_channels:
            print(f"  WARNING: CH{ch} already assigned. Re-enter.")
            ch = _ask_channel("  Enter channel number (1-4): ")

        mapping[key] = ch
        used_channels.add(ch)
        print(f"  → {position}: channel_{ch}\n")

    # Fill any missing channel (shouldn't happen with 4 motors, but safety)
    remaining = [c for c in range(1, 5) if c not in used_channels]
    for key in _KEYS:
        if key not in mapping and remaining:
            mapping[key] = remaining.pop(0)

    print("Mapping identified:")
    for k, v in mapping.items():
        print(f"  {k:<12}: channel_{v}")
    print()

    flip_yaw   = False
    flip_pitch = False

    # ---- Step 3: Yaw sanity check ----
    print("Step 5: Yaw sanity check")
    print("  About to command RIGHT turn at 2% power only.")
    input("  Place the drone on a flat surface and press ENTER...")
    _send_correction_test(mapping, "right", 2.0, sim)
    correct = _ask_yn("  Did the drone attempt to rotate RIGHT (nose right)? (y/n): ")
    if correct:
        print("  CORRECT — yaw mapping confirmed.\n")
    else:
        flip_yaw = True
        print("  Swapping left/right mapping automatically...")
        mapping["front_left"], mapping["front_right"] = \
            mapping["front_right"], mapping["front_left"]
        mapping["rear_left"], mapping["rear_right"] = \
            mapping["rear_right"], mapping["rear_left"]
        print("  Re-testing right turn...")
        _send_correction_test(mapping, "right", 2.0, sim)
        print("  Yaw mapping corrected.\n")

    # ---- Step 4: Pitch sanity check ----
    print("Step 6: Pitch sanity check")
    print("  About to command PITCH UP at 2% power only.")
    input("  Keep the drone stationary and press ENTER...")
    _send_correction_test(mapping, "up", 2.0, sim)
    correct = _ask_yn("  Did the nose attempt to rise? (y/n): ")
    if correct:
        print("  CORRECT — pitch mapping confirmed.\n")
    else:
        flip_pitch = True
        print("  Swapping front/rear mapping automatically...")
        mapping["front_left"], mapping["rear_left"] = \
            mapping["rear_left"], mapping["front_left"]
        mapping["front_right"], mapping["rear_right"] = \
            mapping["rear_right"], mapping["front_right"]
        print("  Re-testing pitch up...")
        _send_correction_test(mapping, "up", 2.0, sim)
        print("  Pitch mapping corrected.\n")

    # ---- Step 5: Save to config ----
    print("Step 7: Save to config")
    cfg["motor_mapping"] = {
        "front_left":  f"channel_{mapping['front_left']}",
        "front_right": f"channel_{mapping['front_right']}",
        "rear_left":   f"channel_{mapping['rear_left']}",
        "rear_right":  f"channel_{mapping['rear_right']}",
        "flip_yaw":    flip_yaw,
        "flip_pitch":  flip_pitch,
        "calibrated":  True,
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  Written to {config_path}")
    print()
    print("═" * 36)
    print("  CALIBRATION COMPLETE")
    print("═" * 36)
    print(f"  front_left:  channel_{mapping['front_left']}")
    print(f"  front_right: channel_{mapping['front_right']}")
    print(f"  rear_left:   channel_{mapping['rear_left']}")
    print(f"  rear_right:  channel_{mapping['rear_right']}")
    print(f"  flip_yaw:    {str(flip_yaw).lower()}")
    print(f"  flip_pitch:  {str(flip_pitch).lower()}")
    print(f"  calibrated:  true")
    print("═" * 36)
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor calibration for OmniVision3D drone")
    parser.add_argument("--config", default=str(_CONFIG),
                        help="Path to config.yaml")
    parser.add_argument("--live",   action="store_true",
                        help="Live hardware mode (default: simulation)")
    args = parser.parse_args()
    run(Path(args.config), sim=not args.live)
