import cv2
import time
import yaml
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vision.yolo_detector import YoloDetector

def main():
    config_path = Path(__file__).parent / 'config.yaml'
    cfg = yaml.safe_load(open(config_path))

    camera_index = 0
    if len(sys.argv) > 1 and sys.argv[1] == '--camera':
        camera_index = int(sys.argv[2])

    print('Starting OmniVision3D...')
    print('Press Q or ESC to quit')

    yolo = YoloDetector(
        model_path=str(Path(__file__).parent / 'vision/yolo_dji.onnx'),
        conf_threshold=0.50,
        input_size=320
    )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {camera_index}')
        return

    window_name = 'OmniVision3D'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name,
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    consecutive = 0

    # Background data (simulated for now, real from sensors later)
    own_lat = 24.7136
    own_lon = 46.6753
    own_compass = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        detections = yolo.detect(frame)

        distance = None
        target_lat = None
        target_lon = None

        if detections:
            best = max(detections, key=lambda d: d.confidence)
            consecutive += 1

            x1, y1, x2, y2 = best.bbox

            box_height = y2 - y1
            if box_height > 0:
                distance = (480 / box_height) * 1.0

            if distance and distance > 0:
                cx = (x1 + x2) / 2
                pixel_offset = cx - (w / 2)
                bearing_offset = (pixel_offset / w) * 60
                target_heading = (own_compass + bearing_offset) % 360

                heading_rad = math.radians(target_heading)
                target_lat = own_lat + (distance * math.cos(heading_rad)) / 111111
                target_lon = own_lon + (distance * math.sin(heading_rad)) / (111111 * math.cos(math.radians(own_lat)))

            color = (0, 255, 0) if consecutive >= 3 else (0, 255, 255)
            thickness = 4 if consecutive >= 3 else 2
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, thickness)
        else:
            consecutive = 0

        if consecutive >= 3:
            cv2.putText(frame, 'DRONE DETECTED',
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            if distance is not None:
                cv2.putText(frame, f'Distance: {distance:.0f} m',
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            if target_lat is not None and target_lon is not None:
                cv2.putText(frame, f'Lat: {target_lat:.6f}',
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Lon: {target_lon:.6f}',
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
        elif consecutive > 0:
            cv2.putText(frame, f'CONFIRMING {consecutive}/3',
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('OmniVision3D stopped')


if __name__ == '__main__':
    main()
