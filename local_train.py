from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


def main() -> None:
    # Step 1: Extract dataset if not already extracted
    dataset_zip = Path(r"C:\Users\Lenovo\Downloads\Tennis Ball detection.v3-non_augmented-data.yolov8.zip")
    dataset_dir = Path("dataset")

    if not (dataset_dir / "data.yaml").exists():
        print(f"[+] Extracting dataset from {dataset_zip}...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if dataset_zip.exists():
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print("[+] Dataset extracted successfully to ./dataset")
        else:
            print(f"[!] Zip file not found at {dataset_zip}")
            sys.exit(1)

    # Step 2: Locate data.yaml
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        yaml_search = list(dataset_dir.rglob("data.yaml"))
        if yaml_search:
            data_yaml = yaml_search[0]

    print(f"[+] Using dataset config: {data_yaml}")

    # Step 3: Check PyTorch GPU availability
    import torch
    from ultralytics import YOLO

    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"
    print(f"[+] CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")

    # Step 4: High-Resolution Fine-tuning (imgsz=640, batch=32, 30 epochs)
    print("[+] Starting High-Resolution 640px YOLOv8n fine-tuning on RTX 5070 GPU...")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=32,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        patience=15,
        seed=42,
        device=device,
        workers=0
    )

    # Step 5: Export best weights to 320x320 ONNX for fast inference
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[+] Training complete! Loading best weights from {best_pt}...")

    trained_model = YOLO(str(best_pt))
    exported_path = Path(
        trained_model.export(
            format="onnx",
            imgsz=320,
            opset=17,
            simplify=True
        )
    )

    output_path = Path("best.onnx")
    if output_path.resolve() != exported_path.resolve():
        output_path.write_bytes(exported_path.read_bytes())

    print(f"[+] ONNX model ready: {output_path.resolve()}")


if __name__ == "__main__":
    main()
