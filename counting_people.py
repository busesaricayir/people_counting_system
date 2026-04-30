"""
counting_people.py — People counting and tracking with YOLOv3 + Centroid Tracker.

Usage:
    python counting_people.py --input video.mp4 --output output/result.mp4
    python counting_people.py --input 0          # webcam

Forte Bilgi İletişim Teknolojileri A.Ş. Internship · July 2023
"""

import argparse
import cv2
import numpy as np

from utils.centroidtracker  import CentroidTracker
from utils.object_trackable import TrackableObject

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS   = "weights/yolov3.weights"
CONFIG    = "weights/yolov3.cfg"
NAMES     = "weights/coco.names"
CONF_THRESH  = 0.5
NMS_THRESH   = 0.4
SKIP_FRAMES  = 30        # run detector every N frames; use tracker in between
INPUT_WIDTH  = 416
INPUT_HEIGHT = 416
PERSON_CLASS = 0         # COCO class index for "person"


def load_model():
    net     = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layer_names(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


def preprocess(frame, rects, ct, trackable_objects, total_up, total_down, line_y):
    """
    Update tracker with new detections and count line crossings.

    Returns updated (total_up, total_down, trackable_objects).
    """
    objects = ct.update(rects)

    for (obj_id, centroid) in objects.items():
        to = trackable_objects.get(obj_id, None)
        if to is None:
            to = TrackableObject(obj_id, centroid)
        else:
            # Determine direction from centroid history mean vs current
            y_vals = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y_vals)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and centroid[1] < line_y:
                    total_up += 1
                    to.counted = True
                elif direction > 0 and centroid[1] > line_y:
                    total_down += 1
                    to.counted = True

        trackable_objects[obj_id] = to

        # Draw centroid dot and ID
        cv2.circle(frame, tuple(centroid), 4, (255, 255, 255), -1)
        cv2.putText(frame, str(obj_id), (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return total_up, total_down, trackable_objects


def run(args):
    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading YOLOv3...")
    net          = load_model()
    output_names = get_output_layer_names(net)

    with open(NAMES) as f:
        class_names = f.read().strip().split("\n")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input if args.input != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {args.input}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    # Reference line: horizontal at mid-frame
    line_y = frame_h // 2

    ct              = CentroidTracker(max_disappeared=40, max_distance=50)
    trackable_objects = {}
    total_up   = 0
    total_down = 0
    frame_count = 0

    print("Processing... (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Draw reference line
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)

        rects = []

        # ── Run detector every SKIP_FRAMES ────────────────────────────────────
        if frame_count % SKIP_FRAMES == 0:
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT),
                swapRB=True, crop=False
            )
            net.setInput(blob)
            layer_outputs = net.forward(output_names)

            boxes, confidences = [], []
            for output in layer_outputs:
                for detection in output:
                    scores     = detection[5:]
                    class_id   = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == PERSON_CLASS and confidence > CONF_THRESH:
                        cx = int(detection[0] * frame_w)
                        cy = int(detection[1] * frame_h)
                        w  = int(detection[2] * frame_w)
                        h  = int(detection[3] * frame_h)
                        xmin = max(0, cx - w // 2)
                        ymin = max(0, cy - h // 2)
                        boxes.append([xmin, ymin, w, h])
                        confidences.append(float(confidence))

            # Non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    rects.append((x, y, x + w, y + h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── Update tracker & count ────────────────────────────────────────────
        total_up, total_down, trackable_objects = preprocess(
            frame, rects, ct, trackable_objects, total_up, total_down, line_y
        )

        # ── Overlay stats ─────────────────────────────────────────────────────
        info = [
            ("Up",   total_up),
            ("Down", total_down),
            ("Total", total_up + total_down),
        ]
        for i, (label, value) in enumerate(info):
            cv2.putText(frame, f"{label}: {value}",
                        (10, frame_h - ((i + 1) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)
        cv2.imshow("KSTS — People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Up: {total_up}  Down: {total_down}  Total: {total_up + total_down}")
    print(f"Output saved → {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="KSTS — People Counting & Tracking")
    p.add_argument("--input",  default="0",              help="Video path or '0' for webcam")
    p.add_argument("--output", default="output/result.mp4", help="Output video path")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
