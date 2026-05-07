# RUN 
# python YOLO\eval_yolo_v3.py

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = "YOLO/runs/results/weights/best.pt"
DEFAULT_DATA = "DATASET/dataset.yaml"
# Dataset class ids: 0..5 are nodes, 6=arrow, 7=arrow_head
DEFAULT_NODE_CLASSES = [0, 1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO validation v3 (node-only for hybrid pipeline: node by YOLO, arrow by CV)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to dataset yaml")
    parser.add_argument("--imgsz", type=int, default=512, help="Validation image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu, 0")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument(
        "--node-only",
        action="store_true",
        default=True,
        help="Evaluate only node classes (0-5), excluding arrow and arrow_head",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Evaluate all classes (override --node-only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    node_only = args.node_only and not args.all_classes
    classes = DEFAULT_NODE_CLASSES if node_only else None

    print("\n" + "=" * 60)
    print("YOLO Validation v3")
    print("=" * 60)
    print(f"Model: {model_path.resolve()}")
    print(f"Data: {data_path.resolve()}")
    print(f"Mode: {'NODE_ONLY (classes 0-5)' if node_only else 'ALL_CLASSES'}")

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_path),
        task="segment",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=True,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
    )

    print("\n" + "=" * 60)
    print("Validation finished")
    print("=" * 60)

    results_dir = Path(results.save_dir)
    print(f"Results saved to: {results_dir.resolve()}")
    print("\nGenerated files:")
    for file_name in sorted(os.listdir(results_dir)):
        print(f"  - {file_name}")

    print("\nKey plots to check:")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - F1_curve.png, P_curve.png, R_curve.png")
    print("  - PR_curve.png")


if __name__ == "__main__":
    main()