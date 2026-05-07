import argparse
from pathlib import Path

from ultralytics import YOLO

# RUN 
# python YOLO\test_one_image_v2.py DATASET/Test/images/10103.png

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained YOLO model on one image.")
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image you want to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="D:/Workspaces/PROJECT/Thesis - Flowchart/runs/segment/runs/flowchart_seg_v2_exp1/weights/best.pt"  ,
        help="Path to model weights (.pt). If omitted, auto-picks latest runs/**/weights/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/single_test",
        help="Output directory",
    )
    parser.add_argument("--name", type=str, default="predict", help="Output run name")
    return parser.parse_args()


def find_latest_best_pt(root: Path) -> Path:
    candidates = list(root.glob("runs/**/weights/best.pt")) + list(root.glob("**/weights/best.pt"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No best.pt found. Pass --model explicitly, e.g. --model runs/segment/runs/exp/weights/best.pt"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_best_pt(Path("."))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Using model: {model_path.resolve()}")
    print(f"Testing image: {image_path.resolve()}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
    )

    output_dir = Path(args.project) / args.name
    print(f"Saved prediction to: {output_dir.resolve()}")

    if results:
        first = results[0]
        print(f"Detected boxes: {len(first.boxes) if first.boxes is not None else 0}")
        print(f"Detected masks: {len(first.masks) if first.masks is not None else 0}")


if __name__ == "__main__":
    main()
