import argparse
from pathlib import Path

from ultralytics import YOLO

# RUN 
# python YOLO\test_one_image_v3.py DATASET/Test/images/10103.png

DEFAULT_MODEL = "YOLO/runs/results/weights/best.pt"
# Dataset class ids: 0..5 are nodes, 6=arrow, 7=arrow_head
DEFAULT_NODE_CLASSES = [0, 1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO v3 on one image (node-only for hybrid pipeline)."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image you want to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to model weights (.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu, 0")
    parser.add_argument(
        "--node-only",
        action="store_true",
        default=True,
        help="Predict only node classes (0-5), excluding arrow and arrow_head",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Predict all classes (override --node-only)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/single_test",
        help="Output directory",
    )
    parser.add_argument("--name", type=str, default="predict", help="Output run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    node_only = args.node_only and not args.all_classes
    classes = DEFAULT_NODE_CLASSES if node_only else None

    print(f"Using model: {model_path.resolve()}")
    print(f"Testing image: {image_path.resolve()}")
    print(f"Mode: {'NODE_ONLY (classes 0-5)' if node_only else 'ALL_CLASSES'}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        classes=classes,
        save=True,
        project=args.project,
        name=args.name,
    )


    if results:
        first = results[0]
        print(f"Detected boxes: {len(first.boxes) if first.boxes is not None else 0}")
        print(f"Detected masks: {len(first.masks) if first.masks is not None else 0}")


if __name__ == "__main__":
    main()
