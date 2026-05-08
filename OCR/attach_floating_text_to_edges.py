# Script to attach floating texts to edges
# Dùng để gắn các floating text (Yes/No/True/False) vào edge tương ứng
# Cách làm: tính khoảng cách từ text center tới edge polyline
# → attach vào edge gần nhất

# python OCR/attach_floating_text_to_edges.py \
#   --ocr-input runs/ocr_full_post \
#   --graph-input runs/graph_v2 \
#   --output runs/graph_with_labels

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach floating texts to edges based on distance to polylines."
    )
    parser.add_argument(
        "--ocr-input",
        type=str,
        required=True,
        help="Input OCR JSON file or directory (with floating_texts)",
    )
    parser.add_argument(
        "--graph-input",
        type=str,
        required=True,
        help="Input graph JSON file or directory (with edges)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for combined JSON",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=50.0,
        help="Max distance to consider attaching text to edge",
    )
    return parser.parse_args()


def distance_point_to_segment(point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate minimum distance from point to line segment"""
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Parameter t of the projection
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def distance_point_to_polyline(point: tuple[float, float], polyline: list[list[float]]) -> float:
    """Calculate minimum distance from point to polyline"""
    if len(polyline) < 2:
        return float('inf')
    
    min_dist = float('inf')
    for i in range(len(polyline) - 1):
        p1 = tuple(polyline[i])
        p2 = tuple(polyline[i + 1])
        dist = distance_point_to_segment(point, p1, p2)
        min_dist = min(min_dist, dist)
    
    return min_dist


def attach_floating_texts_to_edges(
    floating_texts: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    distance_threshold: float,
) -> list[dict[str, Any]]:
    """Attach floating texts to nearest edges"""
    
    # Build node id -> node bbox mapping
    node_bboxes = {node["node_id"]: node["bbox_xyxy"] for node in nodes}
    
    attached_texts = []
    
    for text in floating_texts:
        text_center = tuple(text["center"])
        
        best_edge_idx = -1
        best_distance = float('inf')
        
        for edge_idx, edge in enumerate(edges):
            # Get edge polyline (simplified by connecting source and target node centers)
            source_node_id = edge.get("source")
            target_node_id = edge.get("target")
            
            if source_node_id not in node_bboxes or target_node_id not in node_bboxes:
                continue
            
            source_bbox = node_bboxes[source_node_id]
            target_bbox = node_bboxes[target_node_id]
            
            # Calculate center of nodes
            source_center = [
                (source_bbox[0] + source_bbox[2]) / 2,
                (source_bbox[1] + source_bbox[3]) / 2,
            ]
            target_center = [
                (target_bbox[0] + target_bbox[2]) / 2,
                (target_bbox[1] + target_bbox[3]) / 2,
            ]
            
            # Use polyline if available, otherwise use simple line
            if "polyline" in edge and edge["polyline"]:
                polyline = edge["polyline"]
            else:
                polyline = [source_center, target_center]
            
            # Calculate distance
            dist = distance_point_to_polyline(text_center, polyline)
            
            if dist < best_distance:
                best_distance = dist
                best_edge_idx = edge_idx
        
        # Attach to best edge if within threshold
        if best_edge_idx >= 0 and best_distance <= distance_threshold:
            attached_text = text.copy()
            attached_text["attached_to_edge_idx"] = best_edge_idx
            attached_text["distance_to_edge"] = best_distance
            attached_texts.append(attached_text)
        else:
            # Not attached
            attached_text = text.copy()
            attached_text["attached_to_edge_idx"] = -1
            attached_text["distance_to_edge"] = best_distance
            attached_texts.append(attached_text)
    
    return attached_texts


def load_json_file(path: Path) -> dict[str, Any]:
    """Load JSON file"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    """Save JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def process_one_image(
    ocr_json_path: Path,
    graph_json_path: Path,
    output_path: Path,
    distance_threshold: float,
) -> None:
    """Process one image: combine OCR and graph, attach floating texts"""
    
    # Load OCR data
    ocr_data = load_json_file(ocr_json_path)
    nodes = ocr_data.get("nodes", [])
    floating_texts = ocr_data.get("floating_texts", [])
    
    # Load graph data
    graph_data = load_json_file(graph_json_path)
    edges = graph_data.get("edges", [])
    
    # Attach floating texts to edges
    attached_texts = attach_floating_texts_to_edges(
        floating_texts=floating_texts,
        nodes=nodes,
        edges=edges,
        distance_threshold=distance_threshold,
    )
    
    # Count attached vs unattached
    attached_count = sum(1 for t in attached_texts if t.get("attached_to_edge_idx", -1) >= 0)
    unattached_count = len(attached_texts) - attached_count
    
    # Combine data
    output_data = {
        "image": ocr_data.get("image"),
        "ocr_json": str(ocr_json_path),
        "graph_json": str(graph_json_path),
        "nodes": nodes,
        "edges": edges,
        "floating_texts": attached_texts,
        "floating_text_stats": {
            "total": len(attached_texts),
            "attached": attached_count,
            "unattached": unattached_count,
        },
        "distance_threshold": distance_threshold,
    }
    
    save_json_file(output_path, output_data)
    print(f"✓ {ocr_json_path.stem}: {attached_count} attached, {unattached_count} unattached")


def main() -> None:
    args = parse_args()
    
    ocr_input = Path(args.ocr_input)
    graph_input = Path(args.graph_input)
    output_dir = Path(args.output)
    
    if not ocr_input.exists():
        raise FileNotFoundError(f"OCR input not found: {ocr_input}")
    if not graph_input.exists():
        raise FileNotFoundError(f"Graph input not found: {graph_input}")
    
    # Handle single file vs directory
    if ocr_input.is_file():
        # Single file mode
        stem = ocr_input.stem.split(".")[0]  # Extract sample ID
        
        # Find corresponding graph JSON
        if graph_input.is_file():
            graph_path = graph_input
        else:
            # Search in graph directory
            graph_files = list(graph_input.glob(f"{stem}*"))
            if not graph_files:
                raise FileNotFoundError(f"No matching graph file for {stem}")
            graph_path = graph_files[0]
        
        output_file = output_dir / ocr_input.name
        process_one_image(
            ocr_json_path=ocr_input,
            graph_json_path=graph_path,
            output_path=output_file,
            distance_threshold=args.distance_threshold,
        )
    else:
        # Directory mode
        ocr_files = sorted(ocr_input.glob("*.json"))
        if not ocr_files:
            raise FileNotFoundError(f"No JSON files in {ocr_input}")
        
        for ocr_file in ocr_files:
            # Extract sample ID
            stem = ocr_file.stem.split(".")[0]
            
            # Find corresponding graph file
            if graph_input.is_file():
                graph_file = graph_input
            else:
                graph_candidates = list(graph_input.glob(f"{stem}*"))
                if not graph_candidates:
                    print(f"✗ {ocr_file.name}: no matching graph file")
                    continue
                graph_file = graph_candidates[0]
            
            output_file = output_dir / ocr_file.name
            try:
                process_one_image(
                    ocr_json_path=ocr_file,
                    graph_json_path=graph_file,
                    output_path=output_file,
                    distance_threshold=args.distance_threshold,
                )
            except Exception as e:
                print(f"✗ {ocr_file.name}: {e}")


if __name__ == "__main__":
    main()
