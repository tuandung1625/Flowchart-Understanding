# Tác dụng: Xử lý OCR cho một hình ảnh duy nhất
    # Nhận input 1 ảnh flowchart, dùng YOLO detect nodes (6 classes)
    # Sử dụng PaddleOCR để nhận diện text trong từng node
    # Xuất kết quả thành JSON file

# RUN:
# python OCR/ocr_nodes_v3.py DATASET/Test/images/10103.png --output runs/ocr

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


DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "start",
    1: "end",
    2: "inputoutput",
    3: "operation",
    4: "subroutine",
    5: "condition",
}
V3_NODE_CLASS_IDS = set(DEFAULT_CLASS_NAMES.keys())
DEFAULT_MODEL_PATH = "YOLO/runs/results/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO node detection then PaddleOCR per node and export JSON."
    )
    parser.add_argument("image", type=str, help="Path to a flowchart image")
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
        help="Optional JSON/YAML-like names string or path to dataset.yaml for class names",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--pad", type=int, default=6, help="Crop padding in pixels")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR language")
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=2.0,
        help="Upscale factor applied to each node crop before OCR",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save preprocessed OCR crops next to the JSON output for debugging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr",
        help="Output JSON path or output folder",
    )
    return parser.parse_args()


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

        text = path.read_text(encoding="utf-8")
        marker = "names:"
        idx = text.find(marker)
        if idx == -1:
            return {}
        names_line = text[idx + len(marker) :].strip()
        if names_line.startswith("[") and names_line.endswith("]"):
            raw = names_line[1:-1].strip()
            if not raw:
                return {}
            items = [item.strip().strip("'\"") for item in raw.split(",")]
            return {i: v for i, v in enumerate(items)}
        return {}

    if names_arg.startswith("[") and names_arg.endswith("]"):
        raw = names_arg[1:-1].strip()
        if not raw:
            return {}
        items = [item.strip().strip("'\"") for item in raw.split(",")]
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

    return DEFAULT_CLASS_NAMES.copy()


def is_text_node_class(class_id: int) -> bool:
    return class_id in V3_NODE_CLASS_IDS


def as_int_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    ix1 = max(0, int(np.floor(x1)) - pad)
    iy1 = max(0, int(np.floor(y1)) - pad)
    ix2 = min(w, int(np.ceil(x2)) + pad)
    iy2 = min(h, int(np.ceil(y2)) + pad)
    return ix1, iy1, ix2, iy2


def crop_with_polygon(image: np.ndarray, polygon: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x, y, bw, bh = cv2.boundingRect(polygon.astype(np.int32))
    x1, y1, x2, y2 = as_int_bbox(x, y, x + bw, y + bh, w, h, pad)

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return roi, (x1, y1, x2, y2)

    shifted = polygon.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1

    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)

    white_bg = np.full_like(roi, 255)
    fg = cv2.bitwise_and(roi, roi, mask=mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
    crop = cv2.add(fg, bg)
    return crop, (x1, y1, x2, y2)


def crop_with_bbox(image: np.ndarray, bbox_xyxy: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    ix1, iy1, ix2, iy2 = as_int_bbox(x1, y1, x2, y2, w, h, pad)
    return image[iy1:iy2, ix1:ix2].copy(), (ix1, iy1, ix2, iy2)


def preprocess_for_ocr(crop: np.ndarray, scale: float) -> np.ndarray:
    if crop.size == 0:
        return crop

    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_ocr_on_crop(ocr: PaddleOCR, crop: np.ndarray) -> tuple[str, float, list[dict[str, Any]]]:
    if crop.size == 0:
        return "", 0.0, []

    result = ocr.predict(crop)
    if not result:
        return "", 0.0, []

    # PaddleOCR 3.x commonly returns OCRResult objects with payload in result[0]['res'].
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

        lines = []
        for idx, text in enumerate(rec_texts):
            txt = str(text).strip()
            if not txt:
                continue

            conf = 0.0
            if idx < len(rec_scores):
                try:
                    conf = float(rec_scores[idx])
                except (TypeError, ValueError):
                    conf = 0.0

            box = []
            if idx < len(polys):
                poly = polys[idx]
                try:
                    box = [[float(p[0]), float(p[1])] for p in poly]
                except Exception:
                    box = []

            if box:
                cx = float(sum(p[0] for p in box) / len(box))
                cy = float(sum(p[1] for p in box) / len(box))
            else:
                cx = float(idx)
                cy = float(idx)

            lines.append(
                {
                    "text": txt,
                    "conf": conf,
                    "box": box,
                    "cx": cx,
                    "cy": cy,
                }
            )

        lines.sort(key=lambda ln: (round(ln["cy"] / 10.0), ln["cx"]))
        merged_text = " ".join(ln["text"] for ln in lines)
        mean_conf = float(np.mean([ln["conf"] for ln in lines])) if lines else 0.0
        return merged_text, mean_conf, lines

    if not result[0]:
        return "", 0.0, []

    lines = []
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
            except (TypeError, ValueError):
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
                        except (TypeError, ValueError):
                            conf = 0.0
                else:
                    text = str(second).strip()
                    if len(item) >= 3:
                        try:
                            conf = float(item[2])
                        except (TypeError, ValueError):
                            conf = 0.0

        if box is None or not text:
            continue

        cx = float(sum(p[0] for p in box) / 4.0)
        cy = float(sum(p[1] for p in box) / 4.0)
        lines.append(
            {
                "text": text.strip(),
                "conf": float(conf),
                "box": [[float(p[0]), float(p[1])] for p in box],
                "cx": cx,
                "cy": cy,
            }
        )

    lines = [ln for ln in lines if ln["text"]]
    lines.sort(key=lambda ln: (round(ln["cy"] / 10.0), ln["cx"]))

    merged_text = " ".join(ln["text"] for ln in lines)
    mean_conf = float(np.mean([ln["conf"] for ln in lines])) if lines else 0.0
    return merged_text, mean_conf, lines


def resolve_output_path(output_arg: str, image_path: Path) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{image_path.stem}.ocr.json"


def resolve_output_dir(output_arg: str) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        return out.parent
    return out


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
        ocr = PaddleOCR(
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except TypeError:
        ocr = PaddleOCR(lang=args.lang)

    names_from_model = res.names if isinstance(res.names, dict) else {}
    class_names = resolve_class_names(args.names, names_from_model)

    detections: list[dict[str, Any]] = []
    boxes = res.boxes
    masks = res.masks
    output_dir = resolve_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if boxes is None or len(boxes) == 0:
        output_path = resolve_output_path(args.output, image_path)
        payload = {
            "image": str(image_path),
            "model": str(model_path),
            "nodes": [],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"No detections. JSON written to: {output_path}")
        return

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
        if not is_text_node_class(class_id):
            continue

        class_name = class_names.get(class_id, f"class_{class_id}")

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
        "model": str(model_path),
        "node_count": len(detections),
        "nodes": detections,
    }

    output_path = resolve_output_path(args.output, image_path)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OCR JSON written to: {output_path}")
    print(f"Nodes processed: {len(detections)}")


if __name__ == "__main__":
    main()