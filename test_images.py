from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

from inference import create_session, decode_output, preprocess, Detection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images containing detected balls with metric overlays.")
    parser.add_argument("--model", default="best.onnx", help="Path to exported YOLO ONNX model.")
    parser.add_argument("--input", default="dataset/test/images", help="Directory containing test images.")
    parser.add_argument("--labels", default="dataset/test/labels", help="Directory containing YOLO label txt files (optional).")
    parser.add_argument("--output", default="output_test_results", help="Directory to save annotated output images.")
    parser.add_argument("--max-saved", type=int, default=30, help="Maximum number of detected images to save (0 for all).")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument("--clear-output", action="store_true", default=True, help="Clear output folder before saving.")
    return parser.parse_args()


def load_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, cx_norm, cy_norm, width_norm, height_norm = parts[:5]
        cx = float(cx_norm) * image_width
        cy = float(cy_norm) * image_height
        w = float(width_norm) * image_width
        h = float(height_norm) * image_height

        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - h / 2))
        x2 = int(min(image_width - 1, cx + w / 2))
        y2 = int(min(image_height - 1, cy + h / 2))
        boxes.append((x1, y1, x2, y2))

    return boxes


def calculate_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
    return max(0.0, float(iou))


def draw_rich_metrics_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    gt_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw Ground Truth Boxes if present (Blue dashed rectangle / cyan)
    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = gt
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 191, 0), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "GT Ball",
            (gx1, max(15, gy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 191, 0),
            1,
            cv2.LINE_AA,
        )

    # Draw Detections (Bright Green bounding boxes & text badge)
    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det.box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2, cv2.LINE_AA)

        # Find best matching GT IoU if ground truth exists
        best_iou = 0.0
        for gt in gt_boxes:
            best_iou = max(best_iou, calculate_iou(det.box, gt))

        # Badge text
        badge_text = f"Ball #{idx} | Conf: {det.score * 100:.1f}%"
        if gt_boxes:
            badge_text += f" | IoU: {best_iou:.2f}"

        # Draw filled label background box
        (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 8, ty + baseline), (0, 200, 0), -1)
        cv2.putText(
            annotated,
            badge_text,
            (x1 + 4, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw center crosshair point
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Bounding box metric callout on bottom right of box
        metrics_subtext = f"Pos:({cx},{cy}) Size:{bw}x{bh}px"
        cv2.putText(
            annotated,
            metrics_subtext,
            (x1, y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top Metrics Dashboard Banner
    banner_height = 40
    overlay_panel = annotated[0:banner_height, 0:w].copy()
    cv2.rectangle(annotated, (0, 0), (w, banner_height), (30, 30, 30), -1)
    
    info_str = f"DETECTION METRICS | Total Balls Detected: {len(detections)}"
    if gt_boxes:
        info_str += f" | Ground Truth: {len(gt_boxes)}"
    
    cv2.putText(
        annotated,
        info_str,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if len(detections) > 0 else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    labels_path = Path(args.labels)
    output_dir = Path(args.output)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    else:
        print(f"Error: {input_path} does not exist.")
        return

    print(f"Scanning dataset images to extract ONLY images with ball detections...")
    session = create_session(args.model, threads=4)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    saved_count = 0
    total_scanned = 0

    for path in image_paths:
        total_scanned += 1
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        tensor = preprocess(frame, args.imgsz)
        raw_output = session.run([output_name], {input_name: tensor})[0]
        detections = decode_output(raw_output, frame.shape, args.imgsz, args.conf, args.iou)

        # STRICT FILTER: Only save if at least 1 ball detection is present!
        if len(detections) == 0:
            continue

        gt_boxes = load_ground_truth(labels_path / f"{path.stem}.txt", width, height)
        annotated_frame = draw_rich_metrics_overlay(frame, detections, gt_boxes)

        saved_count += 1
        out_file = output_dir / f"detected_{saved_count:03d}_{path.name}"
        cv2.imwrite(str(out_file), annotated_frame)
        print(f"  [{saved_count}] Saved: {out_file.name} (Detected: {len(detections)} ball(s), Max Conf: {max(d.score for d in detections)*100:.1f}%)")

        if args.max_saved > 0 and saved_count >= args.max_saved:
            print(f"Reached maximum limit of {args.max_saved} saved images.")
            break

    print("\n" + "=" * 60)
    print(f"SUCCESS: Extracted {saved_count} images containing detected balls.")
    print(f"Scanned total: {total_scanned} images.")
    print(f"Results saved strictly to: {output_dir.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
