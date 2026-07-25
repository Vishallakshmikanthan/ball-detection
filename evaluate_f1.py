from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from inference import create_session, decode_output, preprocess


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Metrics:
    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONNX ball detector F1 on YOLO labels.")
    parser.add_argument("--model", default="best.onnx")
    parser.add_argument("--images", required=True, help="Directory containing test images.")
    parser.add_argument("--labels", required=True, help="Directory containing YOLO txt labels.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    return parser.parse_args()


def load_ground_truth(label_path: Path, image_width: int, image_height: int) -> np.ndarray:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, cx_norm, cy_norm, width_norm, height_norm = parts[:5]
        cx = float(cx_norm) * image_width
        cy = float(cy_norm) * image_height
        width = float(width_norm) * image_width
        height = float(height_norm) * image_height
        boxes.append(
            [
                cx - width / 2,
                cy - height / 2,
                cx + width / 2,
                cy + height / 2,
            ]
        )

    return np.array(boxes, dtype=np.float32)


def iou_matrix(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if len(predicted) == 0 or len(ground_truth) == 0:
        return np.zeros((len(predicted), len(ground_truth)), dtype=np.float32)

    pred = predicted[:, None, :]
    gt = ground_truth[None, :, :]

    x1 = np.maximum(pred[..., 0], gt[..., 0])
    y1 = np.maximum(pred[..., 1], gt[..., 1])
    x2 = np.minimum(pred[..., 2], gt[..., 2])
    y2 = np.minimum(pred[..., 3], gt[..., 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    pred_area = np.maximum(0.0, pred[..., 2] - pred[..., 0]) * np.maximum(0.0, pred[..., 3] - pred[..., 1])
    gt_area = np.maximum(0.0, gt[..., 2] - gt[..., 0]) * np.maximum(0.0, gt[..., 3] - gt[..., 1])
    union = pred_area + gt_area - intersection
    return intersection / np.maximum(union, 1e-9)


def match_predictions(
    predicted_boxes: np.ndarray,
    predicted_scores: np.ndarray,
    ground_truth_boxes: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    if len(predicted_boxes) == 0:
        return 0, 0, len(ground_truth_boxes)
    if len(ground_truth_boxes) == 0:
        return 0, len(predicted_boxes), 0

    order = predicted_scores.argsort()[::-1]
    overlaps = iou_matrix(predicted_boxes, ground_truth_boxes)
    matched_ground_truth: set[int] = set()
    true_positive = 0
    false_positive = 0

    for pred_index in order:
        gt_index = int(np.argmax(overlaps[pred_index]))
        best_iou = overlaps[pred_index, gt_index]
        if best_iou >= iou_threshold and gt_index not in matched_ground_truth:
            true_positive += 1
            matched_ground_truth.add(gt_index)
        else:
            false_positive += 1

    false_negative = len(ground_truth_boxes) - len(matched_ground_truth)
    return true_positive, false_positive, false_negative


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    image_paths = iter_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    session = create_session(args.model, args.threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    totals = {threshold: Metrics(threshold, 0, 0, 0) for threshold in args.thresholds}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        ground_truth = load_ground_truth(label_path, width, height)

        tensor = preprocess(image, args.imgsz)
        output = session.run([output_name], {input_name: tensor})[0]

        for threshold, metric in totals.items():
            detections = decode_output(output, image.shape, args.imgsz, threshold, args.nms_iou)
            predicted_boxes = np.array([detection.box for detection in detections], dtype=np.float32)
            predicted_scores = np.array([detection.score for detection in detections], dtype=np.float32)
            tp, fp, fn = match_predictions(predicted_boxes, predicted_scores, ground_truth, args.iou)
            metric.true_positive += tp
            metric.false_positive += fp
            metric.false_negative += fn

    print("threshold precision recall f1 tp fp fn")
    best = max(totals.values(), key=lambda metric: metric.f1)
    for metric in totals.values():
        print(
            f"{metric.threshold:.2f} "
            f"{metric.precision:.4f} "
            f"{metric.recall:.4f} "
            f"{metric.f1:.4f} "
            f"{metric.true_positive} "
            f"{metric.false_positive} "
            f"{metric.false_negative}"
        )

    print(
        "best "
        f"threshold={best.threshold:.2f} "
        f"precision={best.precision:.4f} "
        f"recall={best.recall:.4f} "
        f"f1={best.f1:.4f}"
    )


if __name__ == "__main__":
    main()
