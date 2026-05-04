import argparse
import json
import re
from pathlib import Path
from typing import Any


IGNORED_CLASSES = {"arrow", "arrow_head"}

# python YOLO/evaluate_ocr.py --ocr-dir runs/ocr_post --flowchart-dir DATASET/Train/flowchart --output runs/ocr_eval/train_summary.json --match class --ignore-case
# python YOLO/evaluate_ocr.py --ocr-dir runs/ocr_post --flowchart-dir DATASET/Train/flowchart --output runs/ocr_eval/train_summary_strict.json --match class
# python YOLO/evaluate_ocr.py --ocr-dir runs/ocr_post --flowchart-dir DATASET/Train/flowchart --output runs/ocr_eval/train_summary_order.json --match order --ignore-case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR JSON outputs against .flowchart ground truth (CER/WER/Exact Match)."
    )
    parser.add_argument(
        "--ocr-dir",
        type=str,
        required=True,
        help="Directory containing OCR JSON outputs from ocr_nodes.py",
    )
    parser.add_argument(
        "--flowchart-dir",
        type=str,
        required=True,
        help="Directory containing .flowchart files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr_eval/summary.json",
        help="Path to output evaluation JSON",
    )
    parser.add_argument(
        "--match",
        type=str,
        choices=["order", "class"],
        default="class",
        help="Node matching strategy: order-based or greedy class-based",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Lowercase text before scoring",
    )
    parser.add_argument(
        "--strip-punct",
        action="store_true",
        help="Remove punctuation before scoring",
    )
    parser.add_argument(
        "--pred-field",
        type=str,
        default="normalized_text",
        choices=["normalized_text", "ordered_text", "ocr_text"],
        help="Prediction text field to evaluate from OCR JSON",
    )
    return parser.parse_args()


def normalize_text(text: str, ignore_case: bool, strip_punct: bool) -> str:
    out = text.strip()
    out = re.sub(r"\s+", " ", out)
    if ignore_case:
        out = out.lower()
    if strip_punct:
        out = re.sub(r"[^\w\s]", "", out)
        out = re.sub(r"\s+", " ", out).strip()
    return out


def parse_flowchart_nodes(flowchart_path: Path) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    for raw in flowchart_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        # Keep only node declarations like "st3=>start: ..." and ignore edges like "st3->io5".
        if "=>" not in line or ":" not in line:
            continue

        left, node_text = line.split(":", 1)
        left = left.strip()
        node_text = node_text.strip()

        if "=>" in left:
            node_id, node_class = left.split("=>", 1)
        elif "->" in left:
            # Fallback for alternate annotation style.
            node_id, node_class = left.split("->", 1)
        else:
            continue

        node_id = node_id.strip()
        node_class = node_class.strip()
        if not node_id or not node_class:
            continue

        nodes.append({"id": node_id, "class_name": node_class, "text": node_text})
    return nodes


def parse_ocr_nodes(ocr_json_path: Path, pred_field: str) -> list[dict[str, str]]:
    payload = json.loads(ocr_json_path.read_text(encoding="utf-8-sig"))
    nodes = payload.get("nodes", [])
    out: list[dict[str, str]] = []
    for idx, node in enumerate(nodes):
        class_name = str(node.get("class_name", "")).strip()
        if class_name in IGNORED_CLASSES:
            continue

        text = str(node.get(pred_field, "") or "")
        if not text:
            # Fallback to raw OCR text if selected field is missing.
            text = str(node.get("ocr_text", "") or "")

        out.append(
            {
                "id": str(node.get("node_id", f"node_{idx + 1}")),
                "class_name": class_name,
                "text": text,
            }
        )
    return out


def levenshtein(seq_a: list[Any], seq_b: list[Any]) -> int:
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i] + [0] * len(seq_b)
        for j, b in enumerate(seq_b, start=1):
            cost = 0 if a == b else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[-1]


def char_distance(gt: str, pred: str) -> tuple[int, int]:
    gt_chars = list(gt)
    pred_chars = list(pred)
    return levenshtein(gt_chars, pred_chars), max(1, len(gt_chars))


def word_distance(gt: str, pred: str) -> tuple[int, int]:
    gt_words = gt.split() if gt else []
    pred_words = pred.split() if pred else []
    return levenshtein(gt_words, pred_words), max(1, len(gt_words))


def pair_nodes_order(gt_nodes: list[dict[str, str]], pred_nodes: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str] | None]]:
    pairs: list[tuple[dict[str, str], dict[str, str] | None]] = []
    for i, gt in enumerate(gt_nodes):
        pred = pred_nodes[i] if i < len(pred_nodes) else None
        pairs.append((gt, pred))
    return pairs


def pair_nodes_class_greedy(gt_nodes: list[dict[str, str]], pred_nodes: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str] | None]]:
    pairs: list[tuple[dict[str, str], dict[str, str] | None]] = []
    used: set[int] = set()

    for gt in gt_nodes:
        pick = None

        for i, pred in enumerate(pred_nodes):
            if i in used:
                continue
            if pred["class_name"] == gt["class_name"]:
                pick = i
                break

        if pick is None:
            for i, _ in enumerate(pred_nodes):
                if i not in used:
                    pick = i
                    break

        if pick is None:
            pairs.append((gt, None))
        else:
            used.add(pick)
            pairs.append((gt, pred_nodes[pick]))

    return pairs


def sample_id_from_json_name(path: Path) -> str:
    stem = path.stem
    # Handles: "1.json", "1.ocr.json", "1.post.ocr.json".
    parts = stem.split(".")
    if not parts:
        return stem
    return parts[0]


def evaluate_one(
    flowchart_path: Path,
    ocr_json_path: Path,
    match_mode: str,
    ignore_case: bool,
    strip_punct: bool,
    pred_field: str,
) -> dict[str, Any]:
    gt_nodes = parse_flowchart_nodes(flowchart_path)
    pred_nodes = parse_ocr_nodes(ocr_json_path, pred_field=pred_field)

    if match_mode == "order":
        pairs = pair_nodes_order(gt_nodes, pred_nodes)
    else:
        pairs = pair_nodes_class_greedy(gt_nodes, pred_nodes)

    char_err_sum = 0
    char_total_sum = 0
    word_err_sum = 0
    word_total_sum = 0
    exact = 0

    details = []

    for gt, pred in pairs:
        gt_text = normalize_text(gt["text"], ignore_case=ignore_case, strip_punct=strip_punct)
        pred_text_raw = pred["text"] if pred is not None else ""
        pred_text = normalize_text(pred_text_raw, ignore_case=ignore_case, strip_punct=strip_punct)

        c_err, c_total = char_distance(gt_text, pred_text)
        w_err, w_total = word_distance(gt_text, pred_text)

        char_err_sum += c_err
        char_total_sum += c_total
        word_err_sum += w_err
        word_total_sum += w_total

        is_exact = int(gt_text == pred_text)
        exact += is_exact

        details.append(
            {
                "gt_id": gt["id"],
                "gt_class": gt["class_name"],
                "gt_text": gt_text,
                "pred_id": "" if pred is None else pred["id"],
                "pred_class": "" if pred is None else pred["class_name"],
                "pred_text": pred_text,
                "char_edit": c_err,
                "char_total": c_total,
                "word_edit": w_err,
                "word_total": w_total,
                "exact": bool(is_exact),
            }
        )

    n = max(1, len(gt_nodes))

    return {
        "sample": flowchart_path.stem,
        "flowchart": str(flowchart_path),
        "ocr_json": str(ocr_json_path),
        "gt_nodes": len(gt_nodes),
        "pred_nodes": len(pred_nodes),
        "match_mode": match_mode,
        "pred_field": pred_field,
        "cer": (char_err_sum / char_total_sum) if char_total_sum else 0.0,
        "wer": (word_err_sum / word_total_sum) if word_total_sum else 0.0,
        "exact_match": exact / n,
        "details": details,
    }


def main() -> None:
    args = parse_args()

    ocr_dir = Path(args.ocr_dir)
    flowchart_dir = Path(args.flowchart_dir)
    output_path = Path(args.output)

    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR dir not found: {ocr_dir}")
    if not flowchart_dir.exists():
        raise FileNotFoundError(f"Flowchart dir not found: {flowchart_dir}")

    ocr_files = sorted(ocr_dir.glob("*.json"))
    if not ocr_files:
        raise FileNotFoundError(f"No JSON files found in: {ocr_dir}")

    reports = []
    missing_flowchart = []

    for ocr_json in ocr_files:
        sample_id = sample_id_from_json_name(ocr_json)
        fc_path = flowchart_dir / f"{sample_id}.flowchart"
        if not fc_path.exists():
            missing_flowchart.append(sample_id)
            continue

        report = evaluate_one(
            flowchart_path=fc_path,
            ocr_json_path=ocr_json,
            match_mode=args.match,
            ignore_case=args.ignore_case,
            strip_punct=args.strip_punct,
            pred_field=args.pred_field,
        )
        reports.append(report)

    if not reports:
        raise RuntimeError("No matched OCR JSON <-> flowchart pairs were found")

    cer_avg = sum(r["cer"] for r in reports) / len(reports)
    wer_avg = sum(r["wer"] for r in reports) / len(reports)
    em_avg = sum(r["exact_match"] for r in reports) / len(reports)

    payload = {
        "num_samples": len(reports),
        "match_mode": args.match,
        "pred_field": args.pred_field,
        "ignore_case": args.ignore_case,
        "strip_punct": args.strip_punct,
        "cer_avg": cer_avg,
        "wer_avg": wer_avg,
        "exact_match_avg": em_avg,
        "missing_flowchart_samples": missing_flowchart,
        "samples": reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluated samples: {len(reports)}")
    print(f"CER avg: {cer_avg:.4f}")
    print(f"WER avg: {wer_avg:.4f}")
    print(f"Exact Match avg: {em_avg:.4f}")
    print(f"Report saved to: {output_path}")

    if missing_flowchart:
        print(f"Skipped (missing flowchart): {len(missing_flowchart)}")


if __name__ == "__main__":
    main()
