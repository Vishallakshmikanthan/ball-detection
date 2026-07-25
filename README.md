# Real-Time Ball Detection

CPU-friendly real-time ball detection using YOLOv8n exported to ONNX at `320x320`.

## Files

- `train.py`: trains YOLOv8n on a YOLO-format ball dataset and exports `best.onnx`.
- `inference.py`: runs webcam inference with ONNX Runtime, NMS, boxes, confidence, and FPS overlay.
- `evaluate_f1.py`: evaluates precision, recall, and F1 across confidence thresholds.
- `requirements.txt`: Python dependencies.
- `best.onnx`: generated after training; not included until `train.py` is run.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Colab, install dependencies directly:

```bash
pip install -r requirements.txt
```

## Dataset

Use a Roboflow or YOLOv8-format sports ball dataset with a `data.yaml` similar to:

```yaml
path: /content/datasets/ball
train: train/images
val: valid/images
test: test/images
names:
  0: ball
```

Aim for varied backgrounds, ball sizes, lighting, and partial occlusion.

## Train And Export

```bash
python train.py --data /content/datasets/ball/data.yaml --epochs 50 --batch 16 --device 0
```

The script fine-tunes from `yolov8n.pt`, trains at `640x640`, then exports `best.onnx` at `320x320` with ONNX opset 17.

For CPU-only training:

```bash
python train.py --data path/to/data.yaml --device cpu
```

## Real-Time Inference

```bash
python inference.py --model best.onnx --camera 0 --conf 0.40 --iou 0.45 --threads 4
```

Controls:

- Press `q` or `Esc` to quit.
- Use `--conf` to tune precision vs recall.
- Use `--threads` to match your CPU core count.

## Evaluate F1

```bash
python evaluate_f1.py --model best.onnx --images path/to/test/images --labels path/to/test/labels
```

The evaluator prints:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`

It sweeps confidence thresholds from `0.20` to `0.60` and reports the best threshold.

## Results

Fill this table after training and testing on your machine.

| Model | Input | Precision | Recall | F1 | FPS | CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOv8n ONNX | 320 | 0.800 | 0.130 | 0.201 | 31.5 | NVIDIA RTX 5070 GPU |

## Optimization Notes

- YOLOv8n is used because its small parameter count and low FLOPs keep CPU inference practical.
- ONNX Runtime graph optimization is enabled in `inference.py`.
- Input size is fixed to `320x320` for faster inference.
- NMS removes duplicate boxes after confidence filtering.
- The best `--conf` value should be chosen using `evaluate_f1.py`, not guessed.
