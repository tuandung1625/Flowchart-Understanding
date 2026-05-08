# Xử lý full-image OCR nhiều ảnh cùng lúc (chỉ khởi tạo model/OCR 1 lần)
# Đọc tất cả các ảnh từ thư mục hoặc file riêng lẻ

# python OCR/ocr_full_image_batch.py DATASET/Train/images \
#   --model runs/segment/runs/flowchart_seg_v2_exp1/weights/best.pt \
#   --output runs/ocr_full_batch

# Chỉ xử lý 100 ảnh đầu tiên
# python OCR/ocr_full_image_batch.py DATASET/Train/images \
#   --model runs/detect/train/weights/best.pt \
#   --max-images 100

# Bắt đầu từ ảnh thứ 50
# python OCR/ocr_full_image_batch.py DATASET/Train/images \
#   --model runs/detect/train/weights/best.pt \
#   --start-index 50

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

from ocr_full_image import (
    bbox_intersection_over_union,
    classify_text,
    extract_node_info,
    parse_names_arg,
    point_in_polygon,
    preprocess_for_ocr,
    resolve_output_dir,
    resolve_output_path,
    run_ocr_on_full_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO+full-image OCR on one image or all images in a directory (single model/OCR init)."
    )
    parser.add_argument("input", type=str, help="Input image path or directory of images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt weights")
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Optional names list or path to dataset.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR language")
    parser.add_argument("--ocr-scale", type=float, default=1.0, help="Upscale factor before OCR")
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr_full_batch",
        help="Output JSON directory (or JSON path if single image)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold to determine if text belongs to a node",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, process only the first N images after sorting",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based start index in the sorted image list",
    )
    parser.add_argument(
        "--only-unprocessed",
        action="store_true",
        help="Process only images that do not already have output JSON",
    )
    return parser.parse_args()


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in sorted(input_path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def _process_one_image(
    image_path: Path,
    model: YOLO,
    ocr: PaddleOCR,
    class_names: dict[int, str],
    args: argparse.Namespace,
) -> tuple[Path, int, int]:
    """Process one image and return (output_path, node_count, text_count)"""
    results = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results")

    res = results[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image with OpenCV: {image_path}")

    names_from_model = res.names if isinstance(res.names, dict) else {}
    effective_names = class_names
    if not effective_names:
        effective_names = {int(k): str(v) for k, v in names_from_model.items()} if names_from_model else {}

    # Extract nodes
    nodes = extract_node_info(res, effective_names)

    # Preprocess image for OCR
    image_for_ocr = preprocess_for_ocr(image, args.ocr_scale)

    # Run full-image OCR
    text_all = run_ocr_on_full_image(ocr, image_for_ocr)

    # Classify texts
    node_texts = []
    floating_texts = []

    for text_item in text_all:
        classification, node_idx = classify_text(text_item, nodes, args.iou_threshold)

        if classification == "node_text" and node_idx >= 0:
            text_item["node_id"] = nodes[node_idx]["node_id"]
            text_item["node_class"] = nodes[node_idx]["class_name"]
            node_texts.append(text_item)
        else:
            floating_texts.append(text_item)

    # Build output payload
    payload = {
        "image": str(image_path),
        "model": str(args.model),
        "image_size": [image.shape[1], image.shape[0]],
        "ocr_scale": args.ocr_scale,
        "iou_threshold": args.iou_threshold,
        "node_count": len(nodes),
        "text_total": len(text_all),
        "node_text_count": len(node_texts),
        "floating_text_count": len(floating_texts),
        "nodes": nodes,
        "text_all": [
            {
                "text": t["text"],
                "conf": t["conf"],
                "bbox_xyxy": t["bbox_xyxy"],
                "center": t["center"],
            }
            for t in text_all
        ],
        "node_texts": [
            {
                "text": t["text"],
                "conf": t["conf"],
                "bbox_xyxy": t["bbox_xyxy"],
                "center": t["center"],
                "node_id": t.get("node_id"),
                "node_class": t.get("node_class"),
            }
            for t in node_texts
        ],
        "floating_texts": [
            {
                "text": t["text"],
                "conf": t["conf"],
                "bbox_xyxy": t["bbox_xyxy"],
                "center": t["center"],
            }
            for t in floating_texts
        ],
    }

    output_path = resolve_output_path(args.output, image_path)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, len(nodes), len(text_all)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    images = _collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No image files found in: {input_path}")

    if args.only_unprocessed:
        images = [img for img in images if not resolve_output_path(args.output, img).exists()]
        if not images:
            print("No unprocessed images found. Nothing to do.")
            return

    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")

    start_idx = args.start_index - 1
    if start_idx >= len(images):
        raise ValueError(
            f"--start-index ({args.start_index}) is out of range for {len(images)} selected images"
        )

    images = images[start_idx:]

    if args.max_images > 0:
        images = images[: args.max_images]

    class_names = parse_names_arg(args.names)

    model = YOLO(str(model_path))
    try:
        ocr = PaddleOCR(
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except TypeError:
        ocr = PaddleOCR(lang=args.lang)

    total = len(images)
    ok = 0
    failed = 0

    for i, img_path in enumerate(images, start=1):
        try:
            out_path, nodes, texts = _process_one_image(
                image_path=img_path,
                model=model,
                ocr=ocr,
                class_names=class_names,
                args=args,
            )
            ok += 1
            print(f"[{i}/{total}] OK {img_path.name} -> {out_path.name} (nodes={nodes}, texts={texts})")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] FAIL {img_path.name}: {exc}")

    print(f"Done. success={ok}, failed={failed}, total={total}")


if __name__ == "__main__":
    main()
