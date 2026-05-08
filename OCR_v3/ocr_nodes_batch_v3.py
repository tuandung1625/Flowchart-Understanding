# Xử lý OCR nhiều ảnh cùng lúc (chỉ khởi tạo model/OCR 1 lần)
# Đọc tất cả các ảnh từ thư mục hoặc file riêng lẻ

# python OCR/ocr_nodes_batch_v3.py DATASET/mini_test --model YOLO/runs/results/weights/best.pt --output runs/ocr_batch

# # Chỉ xử lý 100 ảnh đầu tiên
# python OCR/ocr_nodes_batch_v3.py DATASET/Train/images \
#   --model runs/detect/train/weights/best.pt \
#   --max-images 100

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
from ultralytics import YOLO

from ocr_nodes_v3 import (
    V3_NODE_CLASS_IDS,
    crop_with_bbox,
    crop_with_polygon,
    parse_names_arg,
    preprocess_for_ocr,
    resolve_output_dir,
    resolve_output_path,
    run_ocr_on_crop,
)

DEFAULT_MODEL_PATH = "YOLO/runs/results/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO+OCR on one image or all images in a directory (single model/OCR init)."
    )
    parser.add_argument("input", type=str, help="Input image path or directory of images")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO .pt weights (v3 default: YOLO/runs/results/weights/best.pt)",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Optional names list or path to dataset.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--pad", type=int, default=6, help="Crop padding in pixels")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR language")
    parser.add_argument("--ocr-scale", type=float, default=2.0, help="Upscale factor before OCR")
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save preprocessed OCR crops for debugging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr",
        help="Output JSON directory (or JSON path if single image)",
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
) -> tuple[Path, int]:
    results = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results")

    res = results[0]

    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image with OpenCV: {image_path}")

    names_from_model = res.names if isinstance(res.names, dict) else {}
    effective_names = class_names
    if not effective_names:
        effective_names = {int(k): str(v) for k, v in names_from_model.items()} if names_from_model else {}

    detections: list[dict[str, Any]] = []
    boxes = res.boxes
    masks = res.masks
    output_dir = resolve_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if boxes is None or len(boxes) == 0:
        output_path = resolve_output_path(args.output, image_path)
        payload = {
            "image": str(image_path),
            "model": str(args.model),
            "nodes": [],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path, 0

    xyxy_all = boxes.xyxy.cpu().numpy()
    conf_all = boxes.conf.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)

    polygons = masks.xy if masks is not None else [None] * len(xyxy_all)
    if len(polygons) != len(xyxy_all):
        polygons = [None] * len(xyxy_all)

    order = sorted(range(len(xyxy_all)), key=lambda i: (xyxy_all[i][1], xyxy_all[i][0]))

    for rank, i in enumerate(order, start=1):
        bbox = xyxy_all[i]
        poly = polygons[i]
        class_id = int(cls_all[i])
        if class_id not in V3_NODE_CLASS_IDS:
            continue

        class_name = effective_names.get(class_id, f"class_{class_id}")

        if poly is not None and len(poly) >= 3:
            crop, crop_bbox = crop_with_polygon(image, np.array(poly, dtype=np.float32), pad=args.pad)
            polygon_abs = [[float(p[0]), float(p[1])] for p in poly]
        else:
            crop, crop_bbox = crop_with_bbox(image, bbox, pad=args.pad)
            polygon_abs = []

        crop_for_ocr = preprocess_for_ocr(crop, args.ocr_scale)
        if args.save_crops:
            crop_path = output_dir / f"{image_path.stem}_{rank}.png"
            cv2.imwrite(str(crop_path), crop_for_ocr)

        text, text_conf, ocr_lines = run_ocr_on_crop(ocr, crop_for_ocr)

        node = {
            "node_id": f"node_{rank}",
            "class_id": class_id,
            "class_name": class_name,
            "det_conf": float(conf_all[i]),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "crop_bbox_xyxy": [int(v) for v in crop_bbox],
            "polygon": polygon_abs,
            "ocr_text": text,
            "ocr_conf": text_conf,
            "ocr_lines": [
                {
                    "text": ln["text"],
                    "conf": ln["conf"],
                    "box": ln["box"],
                }
                for ln in ocr_lines
            ],
        }
        detections.append(node)

    payload = {
        "pipeline_version": "v3_6class",
        "image": str(image_path),
        "model": str(args.model),
        "node_count": len(detections),
        "nodes": detections,
    }

    output_path = resolve_output_path(args.output, image_path)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, len(detections)


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
            out_path, nodes = _process_one_image(
                image_path=img_path,
                model=model,
                ocr=ocr,
                class_names=class_names,
                args=args,
            )
            ok += 1
            print(f"[{i}/{total}] OK {img_path.name} -> {out_path.name} (nodes={nodes})")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] FAIL {img_path.name}: {exc}")

    print(f"Done. success={ok}, failed={failed}, total={total}")


if __name__ == "__main__":
    main()
