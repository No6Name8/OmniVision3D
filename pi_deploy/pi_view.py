import cv2
import sys
import math
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vision.yolo_detector import YoloDetector
from ground_station import shared_state as ss
from ground_station import targeting
from ground_station.sensors import gps     as gps_sensor
from ground_station.sensors import laser   as laser_sensor
from ground_station.sensors import compass as compass_sensor


def _start_sensors(args):
    if args.gps:
        threading.Thread(
            target=gps_sensor.thread_gps,
            args=(args.gps,),
            kwargs={"baud": 9600},
            daemon=True,
        ).start()

    if args.laser:
        threading.Thread(
            target=laser_sensor.thread_laser,
            args=(args.laser,),
            kwargs={"period": 0.02, "scale": args.laser_scale},
            daemon=True,
        ).start()

    if args.compass:
        threading.Thread(
            target=compass_sensor.thread_compass,
            args=(args.compass,),
            daemon=True,
        ).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",      type=int,   default=0)
    ap.add_argument("--model",       default="vision/yolo_dji.onnx")
    ap.add_argument("--gps",         default=None)
    ap.add_argument("--laser",       default=None)
    ap.add_argument("--compass",     default=None)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--laser-scale", type=float, default=1.0, dest="laser_scale")
    args = ap.parse_args()

    _start_sensors(args)

    print("Starting OmniVision3D...")
    print("Press Q or ESC to quit")

    yolo = YoloDetector(
        model_path=str(Path(__file__).parent / args.model),
        conf_threshold=0.50,
        input_size=320,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    win = "OmniVision3D"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    consecutive = 0
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        detections = yolo.detect(frame)

        # Update target from real sensors
        targeting.update_target(args.declination)

        with ss.lock():
            distance  = ss.state.distance_m
            target_lat = ss.state.target_lat
            target_lon = ss.state.target_lon

        if detections:
            best = max(detections, key=lambda d: d.confidence)
            consecutive += 1
            x1, y1, x2, y2 = best.bbox

            # Fall back to visual distance if no laser
            if distance is None:
                box_h = y2 - y1
                if box_h > 0:
                    distance = (480 / box_h) * 1.0

            color     = (0, 255, 0) if consecutive >= 3 else (0, 255, 255)
            thickness = 4           if consecutive >= 3 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            consecutive = 0

        # Overlay
        if consecutive >= 3:
            cv2.putText(frame, "DRONE DETECTED",
                        (20, 40), FONT, 1.0, (0, 255, 0), 2)
            if distance is not None:
                cv2.putText(frame, f"Distance: {distance:.1f} m",
                            (20, 80), FONT, 0.7, (0, 255, 0), 2)
            if target_lat is not None and target_lon is not None:
                cv2.putText(frame, f"Lat: {target_lat:.6f}",
                            (20, 110), FONT, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Lon: {target_lon:.6f}",
                            (20, 140), FONT, 0.7, (0, 255, 0), 2)
        elif consecutive > 0:
            cv2.putText(frame, f"CONFIRMING {consecutive}/3",
                        (20, 40), FONT, 0.8, (0, 255, 255), 2)
        else:
            # Always show laser distance even without detection
            if distance is not None:
                cv2.putText(frame, f"Distance: {distance:.1f} m",
                            (20, 40), FONT, 0.7, (0, 200, 200), 2)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("OmniVision3D stopped")


if __name__ == "__main__":
    main()
