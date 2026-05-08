
# python GRAPH/arrow_cv_v3.py DATASET/Train/images/8.png --ocr-json runs/ocr_full_post/8.full.v3.ocr.json --nodes-json runs/ocr_full_post/8.full.v3.ocr.json --output runs/arrow_v3 --debug-dir runs/arrow_v3/debug_8

"""
Fixed arrow/connector detection for flowcharts.
Simplified and debugged version.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Detect connectors via CV (fixed)")
    p.add_argument("image", type=str, help="Input image path")
    p.add_argument("--output", type=str, default="runs/connectors_fixed", help="Output folder")
    p.add_argument("--min-length", type=float, default=30.0, help="Minimum segment length")
    p.add_argument("--nodes-json", type=str, default="", help="OCR JSON with node bboxes")
    p.add_argument("--node-margin", type=int, default=15, help="Margin around nodes to mask")
    p.add_argument("--ocr-json", type=str, default="", help="OCR JSON with text boxes")
    p.add_argument("--text-margin", type=int, default=3, help="Margin around text to mask")
    p.add_argument("--debug-dir", type=str, default="", help="Debug images folder")
    return p.parse_args()


def line_length(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def load_boxes_from_json(json_path: str, key: str = "nodes") -> list:
    """Load bounding boxes from JSON file."""
    if not json_path or not Path(json_path).exists():
        return []
    
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    boxes = []
    
    if key == "nodes":
        for node in data.get("nodes", []):
            bbox = node.get("bbox_xyxy")
            if bbox and len(bbox) == 4:
                boxes.append([float(x) for x in bbox])
    else:  # text boxes
        for text_key in ("text_all", "node_texts", "floating_texts"):
            for item in data.get(text_key, []):
                bbox = item.get("bbox_xyxy")
                if bbox and len(bbox) == 4:
                    boxes.append([float(x) for x in bbox])
    
    return boxes


def create_mask(shape, boxes: list, margin: int) -> np.ndarray:
    """Create mask for given bounding boxes."""
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask


def detect_lines(image: np.ndarray, min_length: float) -> list:
    """Detect lines using HoughLinesP."""
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(min_length),
        maxLineGap=15
    )
    
    if lines is None:
        return []
    
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = (float(x1), float(y1))
        p2 = (float(x2), float(y2))
        
        if line_length(p1, p2) >= min_length:
            segments.append((p1, p2))
    
    return segments


def merge_segments(segments: list, max_gap: float = 20.0) -> list:
    """Merge nearby segments into polylines."""
    if not segments:
        return []
    
    # Group connected segments
    groups = []
    used = [False] * len(segments)
    
    for i, (p1, p2) in enumerate(segments):
        if used[i]:
            continue
        
        # Start new group
        group = [p1, p2]
        used[i] = True
        
        # Find connected segments
        changed = True
        while changed:
            changed = False
            for j, (q1, q2) in enumerate(segments):
                if used[j]:
                    continue
                
                # Check if this segment connects to any point in group
                for gp in group:
                    for qp in [q1, q2]:
                        if line_length(gp, qp) <= max_gap:
                            group.extend([q1, q2])
                            used[j] = True
                            changed = True
                            break
                    if changed:
                        break
        
        # Remove duplicates and sort
        unique_pts = []
        for pt in group:
            if pt not in unique_pts:
                unique_pts.append(pt)
        
        if len(unique_pts) >= 2:
            # Sort by x, then y
            unique_pts.sort(key=lambda p: (p[0], p[1]))
            groups.append([[p[0], p[1]] for p in unique_pts])
    
    return groups


def simplify_polyline(points: list, epsilon: float = 5.0) -> list:
    """Simplify polyline using Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon, False)
    
    if approx is None or len(approx) < 2:
        return points
    
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def get_bbox(points: list) -> list:
    """Get bounding box [x1, y1, x2, y2] from points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def get_center(bbox: list) -> list:
    """Get center point from bbox."""
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def main():
    args = parse_args()
    image_path = Path(args.image)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    debug_images = {}
    debug_images["00_original.png"] = gray.copy()
    
    # Load masks
    node_boxes = load_boxes_from_json(args.nodes_json, "nodes")
    text_boxes = load_boxes_from_json(args.ocr_json, "text")
    
    # Create working copy
    work = gray.copy()
    
    # Mask text regions (fill with white)
    if text_boxes:
        text_mask = create_mask((h, w), text_boxes, args.text_margin)
        work[text_mask > 0] = 255
        debug_images["01_text_masked.png"] = work.copy()
    
    # Mask node regions (fill with white)
    if node_boxes:
        node_mask = create_mask((h, w), node_boxes, args.node_margin)
        work[node_mask > 0] = 255
        debug_images["02_nodes_masked.png"] = work.copy()
    
    # Invert: we want black lines on white background -> white lines on black
    inverted = cv2.bitwise_not(work)
    debug_images["03_inverted.png"] = inverted
    
    # Threshold to get clean binary
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    debug_images["04_binary.png"] = binary
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Close small gaps
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    debug_images["05_closed.png"] = closed
    
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    debug_images["06_opened.png"] = opened
    
    # Detect lines using Hough transform
    segments = detect_lines(opened, args.min_length)
    
    print(f"Found {len(segments)} line segments")
    
    # Draw segments for debugging
    if segments:
        seg_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for p1, p2 in segments:
            cv2.line(seg_debug, (int(p1[0]), int(p1[1])), 
                    (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        debug_images["07_segments.png"] = seg_debug
    
    # Merge segments into polylines
    polylines = merge_segments(segments)
    
    print(f"Merged into {len(polylines)} polylines")
    
    # Simplify polylines
    simplified = [simplify_polyline(poly) for poly in polylines]
    
    # Create output
    connectors = []
    for i, poly in enumerate(simplified, start=1):
        if len(poly) < 2:
            continue
        
        bbox = get_bbox(poly)
        center = get_center(bbox)
        
        connectors.append({
            "connector_id": f"c_{i}",
            "polyline": poly,
            "bbox_xyxy": bbox,
            "center": center
        })
    
    # Draw final result
    if connectors:
        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for conn in connectors:
            poly = conn["polyline"]
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            cv2.polylines(result_img, [pts], False, (0, 0, 255), 2)
        debug_images["08_final_result.png"] = result_img
    
    # Save debug images
    if args.debug_dir:
        debug_path = Path(args.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        for name, img in debug_images.items():
            cv2.imwrite(str(debug_path / name), img)
        print(f"Debug images saved to {debug_path}")
    
    # Save output JSON
    output_path = Path(args.output)
    if output_path.suffix != ".json":
        output_path = output_path / f"{image_path.stem}.connectors.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "image": str(image_path),
        "connector_count": len(connectors),
        "connectors": connectors
    }
    
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✓ Detected {len(connectors)} connectors -> {output_path}")


if __name__ == "__main__":
    main()