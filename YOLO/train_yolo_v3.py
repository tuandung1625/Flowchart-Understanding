import argparse
import glob
from pathlib import Path

import torch
from ultralytics import YOLO


def clear_cache_files(data_root: Path) -> None:
    for cache_file in glob.glob(str(data_root / "**" / "*.cache"), recursive=True):
        Path(cache_file).unlink(missing_ok=True)
        print(f"removed cache: {cache_file}")


def verify_train_labels(data_root: Path, train_images_rel: str) -> None:
    image_paths = sorted((data_root / train_images_rel).glob("*.png"))
    matched = 0
    missing = []

    for img_path in image_paths:
        label_path = data_root / "Train" / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            matched += 1
        else:
            missing.append((img_path, label_path))

    print(f"Train images: {len(image_paths)} | matched labels: {matched}")
    print(f"Missing labels: {len(missing)}")

    for img_path, label_path in missing[:30]:
        print(f"IMG: {img_path}")
        print(f"LBL: {label_path}")

    if image_paths and missing:
        raise RuntimeError("Found missing labels in Train split. Fix labels before training.")


def make_runtime_dataset_yaml(
    data_root: Path,
    runs_dir: Path,
    base_yaml: Path,
    train_images_rel: str,
    val_images_rel: str,
    test_images_rel: str,
) -> Path:
    content = base_yaml.read_text(encoding="utf-8").splitlines()
    patched = []

    for line in content:
        stripped = line.strip()
        if stripped.startswith("path:"):
            patched.append(f"path: {data_root.resolve()}")
        elif stripped.startswith("train:"):
            patched.append(f"train: {train_images_rel}")
        elif stripped.startswith("val:"):
            patched.append(f"val: {val_images_rel}")
        elif stripped.startswith("test:"):
            patched.append(f"test: {test_images_rel}")
        else:
            patched.append(line)

    runtime_yaml = runs_dir / "dataset.runtime.yaml"
    runtime_yaml.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print(f"Runtime dataset yaml: {runtime_yaml}")

    return runtime_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation from SSH.")
    parser.add_argument("--data-root", default="./DATASET", help="Dataset root folder")
    parser.add_argument("--data-yaml", default=None, help="Path to dataset.yaml (optional)")
    parser.add_argument("--runs-dir", default="./runs", help="Training output folder")
    parser.add_argument("--train-images", default="Train/images", help="Train images path relative to data root")
    parser.add_argument("--val-images", default="Validation/images", help="Validation images path relative to data root")
    parser.add_argument("--test-images", default="Test/images", help="Test images path relative to data root")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Model checkpoint")
    parser.add_argument("--name", default="flowchart_seg_exp1", help="Run name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs; set 0 to disable periodic checkpoints",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or gpu index like 0")
    parser.add_argument("--predict", action="store_true", help="Run test inference after validation")
    parser.add_argument("--conf", type=float, default=0.25, help="Predict confidence threshold")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_yaml = Path(args.data_yaml) if args.data_yaml else (data_root / "dataset.yaml")
    if not base_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {base_yaml}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    clear_cache_files(data_root)

    verify_train_labels(data_root, args.train_images)

    runtime_yaml = make_runtime_dataset_yaml(
        data_root,
        runs_dir,
        base_yaml,
        args.train_images,
        args.val_images,
        args.test_images,
    )
    device = resolve_device(str(args.device))

    model = YOLO(args.model)
    results = model.train(
        data=str(runtime_yaml),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(runs_dir),
        name=args.name,
        pretrained=True,
        cache=False,
        device=device,
        patience=args.patience,
        save_period=args.save_period,
        save=True,
    )

    print(f"Training complete. Results dir: {runs_dir / args.name}")

    best_weights = runs_dir / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found: {best_weights}")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(runtime_yaml), split="val")
    print(f"Validation metrics: {metrics}")

    if args.predict:
        test_source = data_root / args.test_images
        _ = best_model.predict(
            source=str(test_source),
            imgsz=args.imgsz,
            conf=args.conf,
            save=True,
            project=str(runs_dir),
            name=f"{args.name}_pred",
        )
        print(f"Inference complete. Results at: {runs_dir / f'{args.name}_pred'}")


if __name__ == "__main__":
    main()


# python3 train_yolo.py --data-root ./dataset --runs-dir ./runs --device 0 --batch 32 --epochs 100 --predict

# yolo segment val model=./runs/flowchart_seg_exp1/weights/best.pt data=./DATASET/dataset.yaml split=test imgsz=640