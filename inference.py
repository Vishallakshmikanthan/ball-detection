from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time ball detection from a webcam.")
    parser.add_argument("--model", default="best.onnx", help="Path to exported YOLOv8 ONNX model.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def create_session(model_path: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def preprocess(frame: np.ndarray, image_size: int) -> np.ndarray:
    image = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break

        overlaps = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]

    return keep


def decode_output(
    output: np.ndarray,
    frame_shape: tuple[int, int, int],
    image_size: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected model output shape after squeeze: {predictions.shape}")
    if predictions.shape[0] in (5, 6):
        predictions = predictions.T

    if predictions.shape[1] < 5:
        raise ValueError(f"Expected at least 5 prediction values, got {predictions.shape}")

    scores = predictions[:, 4]
    selected = scores >= confidence_threshold
    if not np.any(selected):
        return []

    predictions = predictions[selected]
    scores = scores[selected]

    frame_height, frame_width = frame_shape[:2]
    scale_x = frame_width / image_size
    scale_y = frame_height / image_size

    cx = predictions[:, 0] * scale_x
    cy = predictions[:, 1] * scale_y
    width = predictions[:, 2] * scale_x
    height = predictions[:, 3] * scale_y

    boxes = np.column_stack(
        (
            np.clip(cx - width / 2, 0, frame_width - 1),
            np.clip(cy - height / 2, 0, frame_height - 1),
            np.clip(cx + width / 2, 0, frame_width - 1),
            np.clip(cy + height / 2, 0, frame_height - 1),
        )
    )

    keep = nms(boxes, scores, iou_threshold)
    return [
        Detection(tuple(int(value) for value in boxes[index]), float(scores[index]))
        for index in keep
    ]


def draw_detections(frame: np.ndarray, detections: list[Detection], fps: float) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"ball {detection.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    session = create_session(args.model, args.threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    fps = 0.0
    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            tensor = preprocess(frame, args.imgsz)
            output = session.run([output_name], {input_name: tensor})[0]
            detections = decode_output(output, frame.shape, args.imgsz, args.conf, args.iou)

            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - last_time, 1e-9)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            last_time = current_time

            draw_detections(frame, detections, fps)
            cv2.imshow("Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
