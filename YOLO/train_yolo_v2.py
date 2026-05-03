"""Train YOLOv8 segmentation model for flowchart detection.

Updated version with 8 classes:
- 0: start
- 1: end
- 2: inputoutput
- 3: operation
- 4: subroutine
- 5: condition
- 6: arrow
- 7: arrow_head

Usage:
    python train_yolo_v2.py --data-root ./DATASET --runs-dir ./runs --device 0 --batch 32 --epochs 100 --predict
"""

import argparse
import glob
from pathlib import Path

import torch
from ultralytics import YOLO


# Updated class mapping matching convert_svg_to_yolo_v2.py
CLASS_MAP = {
    "start": 0,
    "end": 1,
    "inputoutput": 2,
    "operation": 3,
    "subroutine": 4,
    "condition": 5,
    "arrow": 6,
    "arrow_head": 7,
}

CLASS_NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda item: item[1])]


def clear_cache_files(data_root: Path) -> None:
    """Remove all .cache files from dataset directory."""
    for cache_file in glob.glob(str(data_root / "**" / "*.cache"), recursive=True):
        Path(cache_file).unlink(missing_ok=True)
        print(f"Removed cache: {cache_file}")


def verify_train_labels(data_root: Path, train_images_rel: str) -> None:
    """Verify that all training images have corresponding label files."""
    image_paths = sorted((data_root / train_images_rel).glob("*.png"))
    matched = 0
    missing = []

    for img_path in image_paths:
        label_path = data_root / "Train" / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            matched += 1
        else:
            missing.append((img_path, label_path))

    print(f"Train images: {len(image_paths)} | Matched labels: {matched}")
    print(f"Missing labels: {len(missing)}")

    if missing:
        print("\nFirst 30 missing labels:")
        for img_path, label_path in missing[:30]:
            print(f"  IMG: {img_path}")
            print(f"  LBL: {label_path}")

    if image_paths and missing:
        raise RuntimeError(
            f"Found {len(missing)} missing labels in Train split. "
            "Please fix labels before training."
        )


def make_runtime_dataset_yaml(
    data_root: Path,
    runs_dir: Path,
    base_yaml: Path,
    train_images_rel: str,
    val_images_rel: str,
    test_images_rel: str,
) -> Path:
    """Create runtime dataset.yaml with absolute paths and updated class names."""
    content = base_yaml.read_text(encoding="utf-8").splitlines()
    patched = []
    inside_names_section = False

    for line in content:
        stripped = line.strip()
        
        # Update path, train, val, test
        if stripped.startswith("path:"):
            patched.append(f"path: {data_root.resolve()}")
        elif stripped.startswith("train:"):
            patched.append(f"train: {train_images_rel}")
        elif stripped.startswith("val:"):
            patched.append(f"val: {val_images_rel}")
        elif stripped.startswith("test:"):
            patched.append(f"test: {test_images_rel}")
        elif stripped.startswith("nc:"):
            # Update number of classes to 8
            patched.append("nc: 8")
        elif stripped.startswith("names:"):
            # Replace names section with updated class list
            patched.append("names:")
            for class_name in CLASS_NAMES:
                patched.append(f"  - {class_name}")
            inside_names_section = True
        elif inside_names_section:
            # Skip old class names until we hit a non-indented line
            if not line.startswith(" ") and not line.startswith("\t") and stripped:
                inside_names_section = False
                patched.append(line)
        else:
            patched.append(line)

    runtime_yaml = runs_dir / "dataset_v2.runtime.yaml"
    runtime_yaml.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print(f"Runtime dataset yaml: {runtime_yaml}")
    print(f"Classes: {CLASS_NAMES}")

    return runtime_yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation for flowchart detection (8 classes version).")
    parser.add_argument("--data-root",default="./DATASET",help="Dataset root folder")
    parser.add_argument("--data-yaml",default=None,help="Path to dataset.yaml (optional, defaults to <data-root>/dataset.yaml)")
    parser.add_argument("--runs-dir",default="./runs",help="Training output folder")
    parser.add_argument("--train-images",default="Train/images",help="Train images path relative to data root")
    parser.add_argument("--val-images",default="Validation/images",help="Validation images path relative to data root")
    parser.add_argument("--test-images",default="Test/images",help="Test images path relative to data root")
    parser.add_argument("--model",default="yolov8n-seg.pt",help="Model checkpoint (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, etc.)")
    parser.add_argument("--name",default="flowchart_seg_v2_exp1",help="Run name")
    parser.add_argument("--epochs",type=int,default=100,help="Number of training epochs")
    parser.add_argument("--imgsz",type=int,default=640,help="Input image size")
    parser.add_argument("--batch",type=int,default=24,help="Batch size")
    parser.add_argument("--workers",type=int,default=12,help="Number of dataloader workers")
    parser.add_argument("--patience",type=int,default=15,help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--save-period",type=int,default=10,help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--device",default="auto",help="Device: 'auto', 'cpu', or GPU index like '0'")
    parser.add_argument("--predict",action="store_true",help="Run test inference after validation")
    parser.add_argument("--conf",type=float,default=0.25,help="Prediction confidence threshold")
    parser.add_argument("--resume",action="store_true",help="Resume training from last checkpoint")
    return parser.parse_args()


def resolve_device(device_arg: str):
    """Resolve device argument to torch device."""
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def main() -> None:
    """Main training pipeline."""
    args = parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_yaml = Path(args.data_yaml) if args.data_yaml else (data_root / "dataset.yaml")
    if not base_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {base_yaml}")

    # Print system info
    print("=" * 80)
    print("YOLOv8 Training - Flowchart Segmentation (8 Classes)")
    print("=" * 80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Data root: {data_root}")
    print(f"Runs dir: {runs_dir}")
    print(f"Model: {args.model}")
    print(f"Run name: {args.name}")
    print(f"Classes (8): {', '.join(CLASS_NAMES)}")
    print("=" * 80)

    # Cleanup cache files
    clear_cache_files(data_root)

    # Verify training labels
    verify_train_labels(data_root, args.train_images)

    # Create runtime dataset yaml
    runtime_yaml = make_runtime_dataset_yaml(
        data_root,
        runs_dir,
        base_yaml,
        args.train_images,
        args.val_images,
        args.test_images,
    )

    # Resolve device
    device = resolve_device(str(args.device))
    print(f"\nUsing device: {device}")

    # Initialize model
    model = YOLO(args.model)
    
    # Training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
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
        resume=args.resume,
    )

    print("\n" + "=" * 80)
    print(f"Training complete! Results dir: {runs_dir / args.name}")
    print("=" * 80)

    # Validation on best weights
    best_weights = runs_dir / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found: {best_weights}")

    print("\nValidating best model...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(runtime_yaml), split="val", plots=True)
    
    print("\n" + "=" * 80)
    print("Validation Metrics:")
    print("=" * 80)
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    print("=" * 80)

    # Test inference
    if args.predict:
        print("\nRunning test inference...")
        test_source = data_root / args.test_images
        
        predict_results = best_model.predict(
            source=str(test_source),
            imgsz=args.imgsz,
            conf=args.conf,
            save=True,
            project=str(runs_dir),
            name=f"{args.name}_pred",
        )
        
        print(f"\nInference complete! Results: {runs_dir / f'{args.name}_pred'}")
        print(f"Processed {len(predict_results)} images")

    print("\n" + "=" * 80)
    print("All tasks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()


# Example usage:
# Training with prediction:
#   python train_yolo_v2.py --data-root ./DATASET --runs-dir ./runs --device 0 --batch 32 --epochs 100 --predict
#
# Resume training:
#   python train_yolo_v2.py --data-root ./DATASET --runs-dir ./runs --device 0 --resume
#
# Validation only (using best model):
#   yolo segment val model=./runs/flowchart_seg_v2_exp1/weights/best.pt data=./DATASET/dataset.yaml split=val imgsz=640
#
# Test prediction (using best model):
#   yolo segment predict model=./runs/segment/runs/flowchart_seg_v2_exp1/weights/best.pt source=./DATASET/Test/images imgsz=640 conf=0.25 save=True