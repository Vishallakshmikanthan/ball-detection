from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n for ball detection and export a 320px ONNX model."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to YOLO dataset YAML, for example /content/datasets/ball/data.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt", help="Base Ultralytics model.")
    parser.add_argument("--project", default="runs/ball-detection")
    parser.add_argument("--name", default="yolov8n-ball")
    parser.add_argument("--device", default=None, help="Example: 0 for GPU, cpu for CPU.")
    parser.add_argument("--onnx-imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", default="best.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": True,
        "patience": 15,
        "seed": 42,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    trained_model = YOLO(str(best_pt))

    exported_path = Path(
        trained_model.export(
            format="onnx",
            imgsz=args.onnx_imgsz,
            opset=args.opset,
            simplify=True,
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != exported_path.resolve():
        output_path.write_bytes(exported_path.read_bytes())

    print(f"Training complete: {best_pt}")
    print(f"ONNX export ready: {output_path.resolve()}")


if __name__ == "__main__":
    main()
