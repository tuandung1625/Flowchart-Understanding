# Postprocessing script for full-image OCR results
# Làm sạch và cải thiện kết quả OCR
# - Merge node texts từ cùng 1 node
# - Normalize text
# - Filter low confidence items

# python OCR_v3/postprocess_ocr_full.py --input runs/ocr_full_test --output runs/ocr_full_post --normalize

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process full-image OCR JSON results."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input OCR JSON file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory or file path",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Minimum confidence threshold to keep text items",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply text normalization (lowercase, remove extra spaces)",
    )
    parser.add_argument(
        "--merge-node-texts",
        action="store_true",
        default=True,
        help="Merge multiple text items from same node",
    )
    return parser.parse_args()


def _fix_quotes(text: str) -> str:
    """Apply the quote repair rules used by v3."""
    text = re.sub(r':\s*"', r": '", text)
    text = re.sub(r'"\s*\)', r"')", text)
    text = re.sub(r'\(\s*:\s*"', r"(':'", text)
    text = re.sub(r'"\s*if\s+', r"' if ", text)
    text = re.sub(r'\s+else\s+"', r" else '", text)
    return text


def normalize_text(text: str, class_name: str = "") -> str:
    """Normalize text using the same OCR repair rules as v3."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    for _ in range(3):
        updated = text
        updated = re.sub(r"_{2,}", "_", updated)
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
        if updated == text:
            break
        text = updated

    text = _fix_quotes(text)

    if class_name in {"inputoutput", "operation"}:
        text = re.sub(r'\b(input|output):', r'\1: ', text, flags=re.IGNORECASE)

    return text.lower()


def filter_by_confidence(text_items: list[dict[str, Any]], min_conf: float) -> list[dict[str, Any]]:
    """Filter text items by minimum confidence"""
    return [item for item in text_items if item.get("conf", 0.0) >= min_conf]


def merge_node_texts(node_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge multiple texts from the same node in reading order."""
    by_node: dict[str, list[dict[str, Any]]] = {}
    
    for item in node_texts:
        node_id = item.get("node_id", "")
        if node_id not in by_node:
            by_node[node_id] = []
        by_node[node_id].append(item)
    
    merged = []
    for node_id, items in by_node.items():
        if len(items) == 1:
            merged.append(items[0])
        else:
            items.sort(key=lambda x: (x.get("center", [0.0, 0.0])[1], x.get("center", [0.0, 0.0])[0]))

            rows: list[dict[str, Any]] = []
            for item in items:
                center = item.get("center", [0.0, 0.0])
                x = float(center[0]) if len(center) > 0 else 0.0
                y = float(center[1]) if len(center) > 1 else 0.0
                h = max(1.0, float(item.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])[3]) - float(item.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])[1]))

                placed = False
                for row in rows:
                    y_tol = 0.6 * max(row["h"], h)
                    if abs(y - row["y"]) <= y_tol:
                        row["items"].append({"text": str(item.get("text", "")), "x": x, "y": y, "h": h})
                        n = len(row["items"])
                        row["y"] = row["y"] + (y - row["y"]) / n
                        row["h"] = max(row["h"], h)
                        placed = True
                        break
                if not placed:
                    rows.append({"y": y, "h": h, "items": [{"text": str(item.get("text", "")), "x": x, "y": y, "h": h}]})

            rows.sort(key=lambda row: row["y"])
            row_texts: list[str] = []
            for row in rows:
                row["items"].sort(key=lambda w: w["x"])
                row_texts.append(" ".join(word["text"] for word in row["items"]))

            combined_text = " ".join(text for text in row_texts if text).strip()
            mean_conf = sum(float(item.get("conf", 0.0)) for item in items) / len(items)

            merged_item = items[0].copy()
            merged_item["text"] = combined_text
            merged_item["normalized_text"] = normalize_text(combined_text, str(merged_item.get("node_class", "")).strip())
            merged_item["conf"] = mean_conf
            merged_item["merged_from_count"] = len(items)

            merged.append(merged_item)
    
    return merged


def postprocess_one_file(
    input_path: Path,
    output_path: Path,
    min_conf: float,
    normalize: bool,
    merge_node_texts_flag: bool,
) -> None:
    """Post-process a single OCR JSON file"""
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    
    # Process node_texts
    node_texts = payload.get("node_texts", [])
    node_texts = filter_by_confidence(node_texts, min_conf)
    
    if merge_node_texts_flag:
        node_texts = merge_node_texts(node_texts)
    
    if normalize:
        for item in node_texts:
            class_name = str(item.get("node_class", "")).strip()
            item["text"] = normalize_text(item["text"], class_name)
            item["normalized_text"] = item["text"]
    
    # Process floating_texts
    floating_texts = payload.get("floating_texts", [])
    floating_texts = filter_by_confidence(floating_texts, min_conf)
    
    if normalize:
        for item in floating_texts:
            item["text"] = normalize_text(item["text"])
    
    # Update payload
    payload["node_texts"] = node_texts
    payload["floating_texts"] = floating_texts
    payload["node_text_count"] = len(node_texts)
    payload["floating_text_count"] = len(floating_texts)
    payload["text_total"] = len(node_texts) + len(floating_texts)
    payload["postprocess_config"] = {
        "min_conf": min_conf,
        "normalize": normalize,
        "merge_node_texts": merge_node_texts_flag,
    }
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    if input_path.is_file():
        # Single file
        out = output_path if output_path.suffix == ".json" else output_path / input_path.name
        postprocess_one_file(
            input_path=input_path,
            output_path=out,
            min_conf=args.min_conf,
            normalize=args.normalize,
            merge_node_texts_flag=args.merge_node_texts,
        )
        print(f"Processed: {out}")
    else:
        # Directory
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in: {input_path}")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        for json_file in json_files:
            out = output_path / json_file.name
            try:
                postprocess_one_file(
                    input_path=json_file,
                    output_path=out,
                    min_conf=args.min_conf,
                    normalize=args.normalize,
                    merge_node_texts_flag=args.merge_node_texts,
                )
                print(f"✓ {json_file.name}")
            except Exception as e:
                print(f"✗ {json_file.name}: {e}")


if __name__ == "__main__":
    main()
