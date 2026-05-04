# Tác dụng: chỉnh text sau OCR:
    # Sắp xếp lại các từ theo vị trí
    # Chuẩn hóa text (xóa khoảng trắng dư thừa, dấu câu, v.v.)
    # Thêm 2 trường mới vào JSON: ordered_text và normalized_text

# # Xử lý một file JSON
# python OCR/postprocess_ocr.py --input runs/ocr/10103.ocr.json --output runs/ocr_post/10103.post.ocr.json

# # Xử lý toàn bộ thư mục
# python OCR/postprocess_ocr.py --input runs/ocr --output runs/ocr_post

import argparse
import json
import re
from pathlib import Path
from typing import Any


IGNORED_CLASSES = {"arrow", "arrow_head"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process OCR JSON: reorder words by box and normalize per node class."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input OCR JSON file or directory containing *.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr_post",
        help="Output JSON file or directory",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Skip OCR words whose confidence is below this threshold",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file(s) instead of writing to --output",
    )
    return parser.parse_args()


def _clean_spaces(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _move_token_to_front(text: str, token: str) -> str:
    idx = text.lower().find(token.lower())
    if idx < 0:
        return text
    if text[: idx].strip() == "":
        return text

    matched = text[idx : idx + len(token)]
    left = text[:idx]
    right = text[idx + len(token) :]
    tail = _clean_spaces(f"{left} {right}")
    if tail:
        return f"{matched} {tail}"
    return matched


def _repair_code_layout(text: str) -> str:
    out = _clean_spaces(text)

    # Merge identifiers split across OCR lines, especially when underscores or dots are involved.
    for _ in range(3):
        updated = out
        updated = re.sub(
            r"([A-Za-z0-9_.]*[._][A-Za-z0-9_.]*)\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\1_\2",
            updated,
        )
        updated = re.sub(
            r"([A-Za-z0-9_.])\s+_([A-Za-z0-9_]+)",
            r"\1_\2",
            updated,
        )
        updated = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s+\.", r"\1.", updated)
        updated = re.sub(r"\b(input|output)\s*:\s*", r"\1: ", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\s+([,.;:])", r"\1", updated)
        updated = re.sub(r"([([{:])\s+", r"\1", updated)
        updated = re.sub(r"\s+([)\]}])", r"\1", updated)
        updated = re.sub(r"\s+", " ", updated).strip()
        if updated == out:
            break
        out = updated

    return out


def normalize_text(text: str, class_name: str) -> str:
    out = _repair_code_layout(text)

    # Class-agnostic normalization only.
    # Keep semantics unchanged so this is robust for unseen datasets.
    _ = class_name

    return _clean_spaces(out)


def _box_to_rect(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _ordered_text_from_lines(ocr_lines: list[dict[str, Any]], min_conf: float) -> str:
    words = []
    for ln in ocr_lines:
        text = str(ln.get("text", "")).strip()
        conf = float(ln.get("conf", 0.0) or 0.0)
        box = ln.get("box", [])

        if not text or conf < min_conf:
            continue
        if not isinstance(box, list) or len(box) < 2:
            continue

        x1, y1, x2, y2 = _box_to_rect(box)
        words.append(
            {
                "text": text,
                "conf": conf,
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "h": max(1.0, y2 - y1),
            }
        )

    if not words:
        return ""

    words.sort(key=lambda w: (w["y"], w["x"]))

    rows: list[dict[str, Any]] = []
    for w in words:
        placed = False
        for row in rows:
            y_tol = 0.6 * max(row["h"], w["h"])
            if abs(w["y"] - row["y"]) <= y_tol:
                row["items"].append(w)
                n = len(row["items"])
                row["y"] = row["y"] + (w["y"] - row["y"]) / n
                row["h"] = max(row["h"], w["h"])
                placed = True
                break
        if not placed:
            rows.append({"y": w["y"], "h": w["h"], "items": [w]})

    rows.sort(key=lambda r: r["y"])

    row_texts = []
    for row in rows:
        row["items"].sort(key=lambda w: w["x"])
        row_text = " ".join(item["text"] for item in row["items"])
        row_texts.append(_clean_spaces(row_text))

    row_texts = [t for t in row_texts if t]
    if not row_texts:
        return ""

    merged = " ".join(row_texts)
    return _repair_code_layout(merged)


def postprocess_payload(payload: dict[str, Any], min_conf: float) -> dict[str, Any]:
    nodes = payload.get("nodes", [])
    for node in nodes:
        ocr_lines = node.get("ocr_lines", [])
        class_name = str(node.get("class_name", "")).strip()

        if class_name in IGNORED_CLASSES:
            node["ordered_text"] = ""
            node["normalized_text"] = ""
            continue

        ordered_text = _ordered_text_from_lines(ocr_lines, min_conf=min_conf)
        if not ordered_text:
            ordered_text = _clean_spaces(str(node.get("ocr_text", "")))

        normalized_text = normalize_text(ordered_text, class_name)

        node["ordered_text"] = ordered_text
        node["normalized_text"] = normalized_text

    payload["postprocess"] = {
        "version": "1",
        "min_conf": min_conf,
    }
    return payload


def _resolve_outputs(input_path: Path, output_arg: str, in_place: bool) -> Path:
    if in_place:
        return input_path

    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        return out

    out.mkdir(parents=True, exist_ok=True)
    name = input_path.name
    if name.lower().endswith(".ocr.json"):
        name = name[:-9] + ".post.ocr.json"
    else:
        name = input_path.stem + ".post.json"
    return out / name


def process_one_file(input_path: Path, output_path: Path, min_conf: float) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    payload = postprocess_payload(payload, min_conf=min_conf)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Processed: {input_path}")
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    if src.is_file():
        output_path = _resolve_outputs(src, args.output, args.in_place)
        process_one_file(src, output_path, min_conf=args.min_conf)
        return

    files = sorted(src.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in: {src}")

    for file_path in files:
        output_path = _resolve_outputs(file_path, args.output, args.in_place)
        process_one_file(file_path, output_path, min_conf=args.min_conf)


if __name__ == "__main__":
    main()
