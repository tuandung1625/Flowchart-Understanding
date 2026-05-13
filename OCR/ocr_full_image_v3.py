# RUN
# python OCR_v3/ocr_full_image_v3.py DATASET/Train/images/1.png --model YOLO/runs/results/weights/best.pt --names DATASET/dataset_v3.yaml --output runs/ocr_full_test

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from paddleocr import PaddleOCR
from ultralytics import YOLO

DEFAULT_CLASS_NAMES_6 = {
    0: "start",
    1: "end",
    2: "inputoutput",
    3: "operation",
    4: "subroutine",
    5: "condition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO (6-class) node detection then full-image PaddleOCR and classify text (v3)."
    )
    parser.add_argument("image", type=str, help="Path to a flowchart image")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt weights (6-class)")
    parser.add_argument("--names", type=str, default="", help="Optional dataset.yaml or names list")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR language")
    parser.add_argument("--ocr-scale", type=float, default=1.0, help="Upscale factor applied to entire image before OCR")
    parser.add_argument("--output", type=str, default="runs/ocr_full_v3", help="Output JSON path or folder")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold to determine if text belongs to a node")
    return parser.parse_args()


def resolve_output_path(output_arg: str, image_path: Path) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{image_path.stem}.full.v3.ocr.json"


def parse_names_arg(names_arg: str) -> dict[int, str]:
    if not names_arg:
        return {}
    path = Path(names_arg)
    if path.exists() and path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict):
            names = data.get("names", {})
            if isinstance(names, list):
                return {i: str(name).strip() for i, name in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v).strip() for k, v in names.items()}
    if names_arg.startswith("[") and names_arg.endswith("]"):
        raw = names_arg[1:-1].strip()
        if not raw:
            return {}
        items = [item.strip().strip('\"\'') for item in raw.split(",")]
        return {i: v for i, v in enumerate(items)}
    return {}


def resolve_class_names(names_arg: str, model_names: dict[int, str] | None = None) -> dict[int, str]:
    class_names = parse_names_arg(names_arg)
    if class_names:
        return class_names
    if model_names:
        normalized = {int(k): str(v).strip() for k, v in model_names.items()}
        if normalized:
            return normalized
    return DEFAULT_CLASS_NAMES_6.copy()


# Reuse helper functions from original v2 script logic where possible

def bbox_intersection_over_union(box1: list[float], box2: list[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = max(1.0, (x1_max - x1_min)) * max(1.0, (y1_max - y1_min))
    box2_area = max(1.0, (x2_max - x2_min)) * max(1.0, (y2_max - y2_min))
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def run_ocr_on_full_image(ocr: PaddleOCR, image: np.ndarray) -> list[dict[str, Any]]:
    if image.size == 0:
        return []
    result = ocr.predict(image)
    if not result:
        return []
    text_items = []
    first = result[0]
    payload = None
    if isinstance(first, dict):
        payload = first.get("res", first)
    elif hasattr(first, "get"):
        payload = first.get("res", None)
    elif hasattr(first, "res"):
        payload = getattr(first, "res")
    if isinstance(payload, dict) and "rec_texts" in payload:
        rec_texts = payload.get("rec_texts") or []
        rec_scores = payload.get("rec_scores") or []
        polys = payload.get("dt_polys") or payload.get("rec_polys") or []
        for idx, text in enumerate(rec_texts):
            txt = str(text).strip()
            if not txt:
                continue
            conf = 0.0
            if idx < len(rec_scores):
                try:
                    conf = float(rec_scores[idx])
                except Exception:
                    conf = 0.0
            box = []
            if idx < len(polys):
                poly = polys[idx]
                try:
                    box = [[float(p[0]), float(p[1])] for p in poly]
                except Exception:
                    box = []
            if len(box) >= 4:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            else:
                continue
            text_items.append({
                "text": txt,
                "conf": conf,
                "box": box,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
            })
    else:
        if not result[0]:
            return []
        for item in result[0]:
            box = None
            text = ""
            conf = 0.0
            if isinstance(item, dict):
                box = item.get("box") or item.get("dt_polys") or item.get("poly")
                text = str(item.get("text") or item.get("rec_text") or "").strip()
                conf_value = item.get("conf", item.get("score", 0.0))
                try:
                    conf = float(conf_value)
                except Exception:
                    conf = 0.0
            elif isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    box = item[0]
                if len(item) >= 2:
                    second = item[1]
                    if isinstance(second, (list, tuple)):
                        if len(second) >= 1:
                            text = str(second[0]).strip()
                        if len(second) >= 2:
                            try:
                                conf = float(second[1])
                            except Exception:
                                conf = 0.0
                    else:
                        text = str(second).strip()
                        if len(item) >= 3:
                            try:
                                conf = float(item[2])
                            except Exception:
                                conf = 0.0
            if box is None or not text:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            text_items.append({
                "text": text,
                "conf": float(conf),
                "box": [[float(p[0]), float(p[1])] for p in box],
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
            })
    return text_items


def classify_text_item(text_item: dict[str, Any], nodes: list[dict[str, Any]], iou_threshold: float) -> tuple[str, int]:
    text_bbox = text_item["bbox_xyxy"]
    text_center = tuple(text_item["center"])
    best_iou = 0.0
    best_idx = -1
    for idx, node in enumerate(nodes):
        node_bbox = node.get("bbox_xyxy", [])
        if not node_bbox:
            continue
        # If the text center lies inside the node's rectangular bbox,
        # treat it as node text regardless of IoU. This handles small
        # OCR boxes that are fully contained within large node boxes.
        try:
            nx1, ny1, nx2, ny2 = node_bbox
            cx, cy = text_center
            if nx1 <= cx <= nx2 and ny1 <= cy <= ny2:
                return "node_text", idx
        except Exception:
            pass

        iou = bbox_intersection_over_union(text_bbox, node_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
        polygon = node.get("polygon", [])
        if polygon and point_in_polygon(text_center, polygon) and iou >= iou_threshold:
            return "node_text", idx
    if best_iou >= iou_threshold:
        return "node_text", best_idx
    return "floating_text", -1


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results")
    res = results[0]
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image with OpenCV: {image_path}")

    try:
        ocr = PaddleOCR(lang=args.lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)
    except TypeError:
        ocr = PaddleOCR(lang=args.lang)

    names_from_model = res.names if isinstance(res.names, dict) else {}
    class_names = resolve_class_names(args.names, names_from_model)

    # Extract nodes (YOLO detections)
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        output_path = resolve_output_path(args.output, image_path)
        payload = {"image": str(image_path), "model": str(model_path), "nodes": [], "node_texts": [], "floating_texts": []}
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"No detections. JSON written to: {output_path}")
        return

    xyxy_all = boxes.xyxy.cpu().numpy()
    conf_all = boxes.conf.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)

    nodes = []
    for i in range(len(xyxy_all)):
        bbox = xyxy_all[i].tolist()
        class_id = int(cls_all[i])
        class_name = class_names.get(class_id, f"class_{class_id}")
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
            t["node_id"] = nodes[nidx]["node_id"]
            t["node_class"] = nodes[nidx]["class_name"]
            node_texts.append(t)
        else:
            floating_texts.append(t)

    payload = {
        "image": str(image_path),
        "model": str(model_path),
        "node_count": len(nodes),
        "text_total": len(text_all),
        "node_text_count": len(node_texts),
        "floating_text_count": len(floating_texts),
        "nodes": nodes,
        "text_all": [{"text": x["text"], "conf": x["conf"], "bbox_xyxy": x["bbox_xyxy"], "center": x["center"]} for x in text_all],
        "node_texts": [{"text": x["text"], "conf": x["conf"], "bbox_xyxy": x["bbox_xyxy"], "center": x["center"], "node_id": x.get("node_id"), "node_class": x.get("node_class")} for x in node_texts],
        "floating_texts": [{"text": x["text"], "conf": x["conf"], "bbox_xyxy": x["bbox_xyxy"], "center": x["center"]} for x in floating_texts],
    }

    output_path = resolve_output_path(args.output, image_path)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full-image v3 OCR completed: {output_path}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Total text items: {len(text_all)}")
    print(f"  Node text: {len(node_texts)}")
    print(f"  Floating text: {len(floating_texts)}")


if __name__ == "__main__":
    main()
