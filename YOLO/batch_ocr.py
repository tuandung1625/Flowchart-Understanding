import argparse
import json
import sys
from pathlib import Path
from subprocess import run, CalledProcessError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run OCR on all images in a dataset split."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to images directory (e.g., DATASET/Train/images)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model .pt file",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="DATASET/dataset.yaml",
        help="Class names mapping (YAML or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="PaddleOCR language",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=6,
        help="Crop padding for OCR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"No PNG images found in: {images_dir}")
        return

    print(f"Found {len(image_files)} images in: {images_dir}")
    print(f"Output will be saved to: {output_dir}")

    results = []
    succeeded = 0
    failed = 0

    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")

        cmd = [
            sys.executable,
            "YOLO/ocr_nodes.py",
            str(img_path),
            "--model",
            str(args.model),
            "--names",
            str(args.names),
            "--output",
            str(output_dir),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--lang",
            args.lang,
            "--pad",
            str(args.pad),
        ]

        try:
            completed = run(cmd, check=True, capture_output=True, text=True)
            if completed.stdout:
                print(completed.stdout)
            succeeded += 1
            results.append({"image": img_path.name, "status": "ok"})
            print(f"✓ {img_path.name}")
        except CalledProcessError as e:
            failed += 1
            error_text = (e.stderr or e.stdout or str(e)).strip()
            results.append({"image": img_path.name, "status": "error", "error": error_text})
            print(f"✗ {img_path.name}: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

    summary = {
        "total": len(image_files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }

    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Batch OCR complete:")
    print(f"  Total: {len(image_files)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
