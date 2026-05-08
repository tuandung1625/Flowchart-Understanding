# RUN
# python OCR_v3/ocr_full_image_batch_v3.py DATASET/Test/images --model YOLO/runs/results/weights/best.pt --names DATASET/dataset_v3.yaml --output runs/ocr_full_test_batch --max-images 10

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from paddleocr import PaddleOCR
from ultralytics import YOLO

from ocr_full_image_v3 import (
    run_ocr_on_full_image,
    classify_text_item,
    parse_names_arg as parse_names_arg_v3,
    resolve_output_path as resolve_output_path_v3,
    resolve_class_names as resolve_class_names_v3,
)

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch full-image OCR for v3 pipeline")
    parser.add_argument("input", type=str, help="Input image path or directory")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt weights (6-class)")
    parser.add_argument("--names", type=str, default="", help="Optional dataset.yaml or names list")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR language")
    parser.add_argument("--ocr-scale", type=float, default=1.0, help="Upscale factor before OCR")
    parser.add_argument("--output", type=str, default="runs/ocr_full_v3_batch", help="Output directory")
    parser.add_argument("--max-images", type=int, default=0, help="If >0 process only first N images")
    parser.add_argument("--start-index", type=int, default=1, help="1-based start index")
    parser.add_argument("--only-unprocessed", action="store_true", help="Skip images with existing output")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for text->node classification")
    return parser.parse_args()


def _collect_images(path: Path):
    if path.is_file():
        return [path]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _process_one(image_path: Path, model: YOLO, ocr: PaddleOCR, class_names: dict[int, str], args: argparse.Namespace):
    results = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results")
    res = results[0]
    import numpy as np
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    names_from_model = res.names if isinstance(res.names, dict) else {}
    effective_names = class_names or ( {int(k):str(v) for k,v in names_from_model.items()} if names_from_model else {})

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        out = resolve_output_path_v3(args.output, image_path)
        payload = {"image": str(image_path), "model": str(args.model), "nodes": [], "node_texts": [], "floating_texts": []}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out, 0, 0

    xyxy_all = boxes.xyxy.cpu().numpy()
    conf_all = boxes.conf.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)

    nodes = []
    for i in range(len(xyxy_all)):
        bbox = xyxy_all[i].tolist()
        class_id = int(cls_all[i])
        class_name = effective_names.get(class_id, f"class_{class_id}")
        nodes.append({
            "node_id": f"node_{i+1}",
            "class_id": class_id,
            "class_name": class_name,
            "det_conf": float(conf_all[i]),
            "bbox_xyxy": [float(v) for v in bbox],
            "polygon": [],
        })

    image_for_ocr = image.copy()
    if args.ocr_scale != 1.0:
        image_for_ocr = cv2.resize(image_for_ocr, None, fx=args.ocr_scale, fy=args.ocr_scale, interpolation=cv2.INTER_CUBIC)

    text_all = run_ocr_on_full_image(ocr, image_for_ocr)

    node_texts = []
    floating_texts = []
    for t in text_all:
        clsf, nidx = classify_text_item(t, nodes, args.iou_threshold)
        if clsf == "node_text" and nidx >= 0:
            t['node_id'] = nodes[nidx]['node_id']
            t['node_class'] = nodes[nidx]['class_name']
            node_texts.append(t)
        else:
            floating_texts.append(t)

    payload = {
        "image": str(image_path),
        "model": str(args.model),
        "node_count": len(nodes),
        "text_total": len(text_all),
        "node_text_count": len(node_texts),
        "floating_text_count": len(floating_texts),
        "nodes": nodes,
        "text_all": [{"text": x['text'], "conf": x['conf'], "bbox_xyxy": x['bbox_xyxy'], "center": x['center']} for x in text_all],
        "node_texts": [{"text": x['text'], "conf": x['conf'], "bbox_xyxy": x['bbox_xyxy'], "center": x['center'], "node_id": x.get('node_id'), "node_class": x.get('node_class')} for x in node_texts],
        "floating_texts": [{"text": x['text'], "conf": x['conf'], "bbox_xyxy": x['bbox_xyxy'], "center": x['center']} for x in floating_texts],
    }

    out = resolve_output_path_v3(args.output, image_path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, len(nodes), len(text_all)


def main():
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    images = _collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images in: {input_path}")

    if args.only_unprocessed:
        images = [img for img in images if not resolve_output_path_v3(args.output, img).exists()]
        if not images:
            print("No unprocessed images found.")
            return

    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")
    images = images[args.start_index - 1:]
    if args.max_images > 0:
        images = images[:args.max_images]

    class_names = parse_names_arg_v3(args.names)

    model = YOLO(str(model_path))
    try:
        ocr = PaddleOCR(lang=args.lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)
    except TypeError:
        ocr = PaddleOCR(lang=args.lang)

    ok = 0
    failed = 0
    for i, img in enumerate(images, start=1):
        try:
            out, nodes, texts = _process_one(img, model, ocr, class_names, args)
            ok += 1
            print(f"[{i}/{len(images)}] OK {img.name} -> {out.name} (nodes={nodes}, texts={texts})")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] FAIL {img.name}: {e}")
    print(f"Done. success={ok}, failed={failed}, total={len(images)}")

if __name__ == '__main__':
    main()
