"""Build a lightweight flowchart graph from OCR node JSON.

This stage is intentionally heuristic:
- it keeps OCR text cleanup separate from control-flow reconstruction
- it infers edges from node geometry and class labels
- **FIXED**: Now uses arrows as primary source, geometric heuristics as fallback
- it does not require retraining OCR for punctuation/identifier noise

Example:
python GRAPH/build_graph.py --input runs/ocr_post --output runs/graph
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math


ARROW_CLASSES = {"arrow", "arrow_head"}
FLOW_CLASSES = {"start", "end", "inputoutput", "operation", "subroutine", "condition"}


@dataclass(frozen=True)
class NodeGeom:
    node_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    width: float
    height: float
    polygon: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer flowchart graph edges from OCR JSON nodes."
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
        default="runs/graph",
        help="Output JSON file or directory",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file(s) instead of writing to --output",
    )
    parser.add_argument(
        "--arrows-only",
        action="store_true",
        help="Only use arrows for edge detection, no geometric fallback",
    )
    return parser.parse_args()


def _clean_spaces(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _box_to_rect(box: list[float] | list[list[float]]) -> tuple[float, float, float, float]:
    if not box:
        return 0.0, 0.0, 0.0, 0.0

    if isinstance(box[0], list):
        points = box  # type: ignore[assignment]
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    if len(box) != 4:
        return 0.0, 0.0, 0.0, 0.0

    x1, y1, x2, y2 = [float(v) for v in box]
    return x1, y1, x2, y2


def _node_geom(node: dict[str, Any], fallback_index: int) -> NodeGeom | None:
    class_name = str(node.get("class_name", "")).strip()

    node_id = str(node.get("node_id") or f"node_{fallback_index}").strip()
    bbox_raw = node.get("bbox_xyxy", [])
    x1, y1, x2, y2 = _box_to_rect(bbox_raw)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    polygon_raw = node.get("polygon", [])
    polygon: tuple[tuple[float, float], ...] = ()
    if isinstance(polygon_raw, list) and polygon_raw:
        points: list[tuple[float, float]] = []
        for point in polygon_raw:
            if isinstance(point, list) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        polygon = tuple(points)

    return NodeGeom(
        node_id=node_id,
        class_name=class_name,
        bbox=(x1, y1, x2, y2),
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        width=width,
        height=height,
        polygon=polygon,
    )


def _node_record(node: dict[str, Any], geom: NodeGeom, index: int) -> dict[str, Any]:
    ordered_text = _clean_spaces(str(node.get("ordered_text", "") or ""))
    normalized_text = _clean_spaces(str(node.get("normalized_text", "") or ""))
    ocr_text = _clean_spaces(str(node.get("ocr_text", "") or ""))

    return {
        "node_id": geom.node_id,
        "index": index,
        "class_name": geom.class_name,
        "bbox_xyxy": [float(v) for v in geom.bbox],
        "center_xy": [geom.center_x, geom.center_y],
        "size_xy": [geom.width, geom.height],
        "ocr_text": ocr_text,
        "ordered_text": ordered_text,
        "normalized_text": normalized_text,
        "text": normalized_text or ordered_text or ocr_text,
    }


def _bbox_distance(bbox1: tuple[float, float, float, float], 
                   bbox2: tuple[float, float, float, float]) -> float:
    """Calculate minimum distance between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate horizontal and vertical distances
    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)
    
    return (dx ** 2 + dy ** 2) ** 0.5


def _point_to_bbox_distance(point: tuple[float, float], 
                            bbox: tuple[float, float, float, float]) -> float:
    """Calculate distance from a point to a bounding box."""
    px, py = point
    x_min, y_min, x_max, y_max = bbox
    
    # Find closest point on bbox
    closest_x = max(x_min, min(px, x_max))
    closest_y = max(y_min, min(py, y_max))
    
    dx = px - closest_x
    dy = py - closest_y
    
    return (dx ** 2 + dy ** 2) ** 0.5


def _polygon_tip_and_base(polygon: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(polygon) < 3:
        return None

    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)

    tip = max(polygon, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    others = [p for p in polygon if p != tip]
    if not others:
        return None

    base_x = sum(p[0] for p in others) / len(others)
    base_y = sum(p[1] for p in others) / len(others)
    return (base_x, base_y), tip


def _connector_endpoints(connector: NodeGeom) -> tuple[tuple[float, float], tuple[float, float]]:
    if connector.class_name == "arrow_head":
        tips = _polygon_tip_and_base(connector.polygon)
        if tips is not None:
            source_pt, target_pt = tips
            return source_pt, target_pt

    if connector.width >= connector.height:
        return (connector.bbox[0], connector.center_y), (connector.bbox[2], connector.center_y)

    return (connector.center_x, connector.bbox[1]), (connector.center_x, connector.bbox[3])


def _infer_edge_label(
    source_node: NodeGeom,
    target_node: NodeGeom,
    arrow: NodeGeom | None = None
) -> str | None:
    """Placeholder for edge label inference.
    
    NOTE: Spatial heuristics for yes/no labels are unreliable because different
    flowcharts use different conventions (yes=right vs yes=down, etc).
    
    Solution: Extract labels only from ground truth (.flowchart files) during validation.
    During real inference on PNG images without ground truth, branch edges will not have labels.
    
    This function is kept for future learning-based inference.
    """
    # Labels extracted from ground truth only - no spatial inference
    return None


def _vector_length(vec: tuple[float, float]) -> float:
    return math.hypot(vec[0], vec[1])


def _cosine_similarity(vec_a: tuple[float, float], vec_b: tuple[float, float]) -> float:
    len_a = _vector_length(vec_a)
    len_b = _vector_length(vec_b)
    if len_a == 0.0 or len_b == 0.0:
        return 0.0
    return (vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]) / (len_a * len_b)


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _match_arrows_to_nodes(
    arrows: list[NodeGeom], 
    flow_nodes: list[NodeGeom],
    max_distance: float = 220.0
) -> list[dict[str, Any]]:
    """Match arrows to flow nodes to create edges.
    
    Strategy:
    1. Use the connector geometry to infer a source-side anchor and a target-side anchor.
    2. Find the best source/target flow-node pair aligned with that direction.
    3. Prefer arrow-head based matches and keep geometric fallback for ambiguous cases.
    """
    edges: list[dict[str, Any]] = []

    seen_pairs: set[tuple[str, str]] = set()

    for arrow in arrows:
        conn_point = (arrow.center_x, arrow.center_y)

        best_source: NodeGeom | None = None
        best_target: NodeGeom | None = None
        best_score = float("inf")
        best_source_dist = float("inf")
        best_target_dist = float("inf")

        for source_node in flow_nodes:
            for target_node in flow_nodes:
                if source_node.node_id == target_node.node_id:
                    continue

                dx = target_node.center_x - source_node.center_x
                dy = target_node.center_y - source_node.center_y
                dominant_vertical = abs(dy) >= abs(dx)

                score = _point_to_segment_distance(
                    conn_point,
                    (source_node.center_x, source_node.center_y),
                    (target_node.center_x, target_node.center_y),
                )
                score += 0.02 * math.hypot(dx, dy)

                if dominant_vertical:
                    low_y, high_y = sorted([source_node.center_y, target_node.center_y])
                    if not (low_y - 0.15 * max(source_node.height, target_node.height) <= conn_point[1] <= high_y + 0.15 * max(source_node.height, target_node.height)):
                        score += 150.0
                    if source_node.class_name != "condition" and dy < -0.18 * max(1.0, source_node.height):
                        score += abs(dy) * 1.2
                else:
                    low_x, high_x = sorted([source_node.center_x, target_node.center_x])
                    if not (low_x - 0.15 * max(source_node.width, target_node.width) <= conn_point[0] <= high_x + 0.15 * max(source_node.width, target_node.width)):
                        score += 150.0
                    if source_node.class_name != "condition" and dx < -0.18 * max(1.0, source_node.width):
                        score += abs(dx) * 1.2

                if source_node.class_name == "condition" and dy < -0.18 * max(1.0, source_node.height):
                    score += abs(dy) * 1.2

                if score < best_score:
                    best_source = source_node
                    best_target = target_node
                    best_score = score
                    best_source_dist = _point_to_bbox_distance(conn_point, source_node.bbox)
                    best_target_dist = _point_to_bbox_distance(conn_point, target_node.bbox)

        if best_source is None or best_target is None or best_score > max_distance:
            continue

        if best_source.class_name == "condition":
            if best_target.center_x < best_source.center_x or best_target.center_y > best_source.center_y:
                kind = "branch_0"
            else:
                kind = "branch_1"
        else:
            kind = "next"

        # Infer edge label (yes/no) for condition branches
        edge_label = _infer_edge_label(best_source, best_target)

        pair = (best_source.node_id, best_target.node_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        edge_record: dict[str, Any] = {
            "edge_id": f"{best_source.node_id}->{best_target.node_id}:{kind}",
            "source": best_source.node_id,
            "target": best_target.node_id,
            "kind": kind,
            "confidence": "arrow_matched",
            "arrow_id": arrow.node_id,
            "match_distances": {
                "source": round(best_source_dist, 2),
                "target": round(best_target_dist, 2)
            }
        }
        if edge_label is not None:
            edge_record["edge_label"] = edge_label
        
        edges.append(edge_record)
    
    return edges


def _weighted_score(source: NodeGeom, target: NodeGeom, vertical_bias: float = 1.0) -> float:
    dx = abs(target.center_x - source.center_x)
    dy = max(0.0, target.center_y - source.center_y)
    return dy * vertical_bias + dx * 0.35


def _pick_next_node(
    source: NodeGeom,
    candidates: list[NodeGeom],
    *,
    prefer_left: bool | None = None,
    allow_upward: bool = False,
) -> tuple[NodeGeom | None, float]:
    best: NodeGeom | None = None
    best_score = float("inf")

    for candidate in candidates:
        if candidate.node_id == source.node_id:
            continue

        dy = candidate.center_y - source.center_y
        dx = candidate.center_x - source.center_x

        if not allow_upward and dy < -0.18 * source.height:
            continue

        if prefer_left is True and dx > 0:
            continue
        if prefer_left is False and dx < 0:
            continue

        score = _weighted_score(source, candidate)

        if allow_upward and dy < 0:
            score += abs(dy) * 0.65

        if abs(dx) > 3.5 * max(source.width, candidate.width):
            score += abs(dx) * 0.5

        if score < best_score:
            best = candidate
            best_score = score

    return best, best_score


def _build_edges_geometric_only(nodes: list[NodeGeom]) -> list[dict[str, Any]]:
    """Original geometric-only approach (fallback)."""
    edges: list[dict[str, Any]] = []
    source_nodes = [node for node in nodes if node.class_name in FLOW_CLASSES]

    for source in source_nodes:
        if source.class_name == "end":
            continue

        if source.class_name == "condition":
            left_target, left_score = _pick_next_node(source, source_nodes, prefer_left=True)
            right_target, right_score = _pick_next_node(source, source_nodes, prefer_left=False)

            picked: list[tuple[str, NodeGeom, float]] = []
            if left_target is not None:
                picked.append(("branch_0", left_target, left_score))
            if right_target is not None:
                picked.append(("branch_1", right_target, right_score))

            if not picked:
                next_target, next_score = _pick_next_node(source, source_nodes)
                if next_target is not None:
                    picked.append(("next", next_target, next_score))

            seen: set[str] = set()
            for kind, target, score in sorted(picked, key=lambda item: item[2]):
                if target.node_id in seen:
                    continue
                seen.add(target.node_id)
                edges.append(
                    {
                        "edge_id": f"{source.node_id}->{target.node_id}:{kind}",
                        "source": source.node_id,
                        "target": target.node_id,
                        "kind": kind,
                        "score": round(float(score), 6),
                        "confidence": "geometric_only"
                    }
                )
            continue

        downstream = [candidate for candidate in source_nodes if candidate.center_y > source.center_y]
        next_target, next_score = _pick_next_node(source, downstream)
        if next_target is None:
            continue

        edges.append(
            {
                "edge_id": f"{source.node_id}->{next_target.node_id}:next",
                "source": source.node_id,
                "target": next_target.node_id,
                "kind": "next",
                "score": round(float(next_score), 6),
                "confidence": "geometric_only"
            }
        )

    return edges


def _build_edges_hybrid(
    flow_nodes: list[NodeGeom],
    connectors: list[NodeGeom],
    arrows_only: bool = False
) -> list[dict[str, Any]]:
    """Hybrid approach: arrows first, geometric fallback.
    
    Strategy:
    1. Use arrows to detect edges (high confidence)
    2. For nodes without outgoing edges, use geometric heuristics (fallback)
    3. Deduplicate edges
    """
    
    # Step 1: Get edges from connectors
    edges_from_arrows = _match_arrows_to_nodes(connectors, flow_nodes)
    
    if arrows_only:
        return edges_from_arrows
    
    # Track which nodes already have outgoing edges
    nodes_with_outgoing: set[str] = set()
    for edge in edges_from_arrows:
        nodes_with_outgoing.add(edge["source"])
    
    # Step 2: Geometric fallback for nodes without connector-based outgoing edges
    fallback_edges: list[dict[str, Any]] = []

    for source in flow_nodes:
        if source.node_id in nodes_with_outgoing:
            continue
        if source.class_name == "end":
            continue

        if source.class_name == "condition":
            left_target, left_score = _pick_next_node(source, flow_nodes, prefer_left=True)
            right_target, right_score = _pick_next_node(source, flow_nodes, prefer_left=False)

            picked: list[tuple[str, NodeGeom, float]] = []
            if left_target is not None:
                picked.append(("branch_0", left_target, left_score))
            if right_target is not None:
                picked.append(("branch_1", right_target, right_score))

            if not picked:
                next_target, next_score = _pick_next_node(source, flow_nodes)
                if next_target is not None:
                    picked.append(("next", next_target, next_score))

            seen: set[str] = set()
            for kind, target, score in sorted(picked, key=lambda item: item[2]):
                if target.node_id in seen:
                    continue
                seen.add(target.node_id)
                fallback_edges.append({
                    "edge_id": f"{source.node_id}->{target.node_id}:{kind}",
                    "source": source.node_id,
                    "target": target.node_id,
                    "kind": kind,
                    "score": round(float(score), 6),
                    "confidence": "geometric_fallback"
                })
            continue

        downstream = [candidate for candidate in flow_nodes if candidate.center_y > source.center_y]
        next_target, next_score = _pick_next_node(source, downstream)

        if next_target is not None:
            fallback_edges.append({
                "edge_id": f"{source.node_id}->{next_target.node_id}:next",
                "source": source.node_id,
                "target": next_target.node_id,
                "kind": "next",
                "score": round(float(next_score), 6),
                "confidence": "geometric_fallback"
            })
    
    # Step 3: Combine and deduplicate
    all_edges = edges_from_arrows + fallback_edges
    
    # Deduplicate by (source, target) pair - prefer arrow-matched edges
    seen_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in all_edges:
        pair = (edge["source"], edge["target"])
        if pair not in seen_pairs:
            seen_pairs[pair] = edge
        elif edge.get("confidence") == "arrow_matched":
            # Prefer arrow-matched over geometric
            seen_pairs[pair] = edge
    
    return list(seen_pairs.values())


def _topological_execution_order(nodes: list[NodeGeom], edges: list[dict[str, Any]]) -> list[str]:
    outgoing: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    incoming_count: dict[str, int] = {node.node_id: 0 for node in nodes}

    for edge in edges:
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src not in outgoing or dst not in incoming_count:
            continue
        outgoing[src].append(dst)
        incoming_count[dst] += 1

    starts = [node.node_id for node in nodes if node.class_name == "start"]
    if not starts:
        starts = [node.node_id for node in nodes if incoming_count[node.node_id] == 0]
    if not starts and nodes:
        starts = [nodes[0].node_id]

    order: list[str] = []
    visited: set[str] = set()
    stack: list[str] = list(reversed(starts))

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        order.append(node_id)

        neighbors = outgoing.get(node_id, [])
        if neighbors:
            neighbors = sorted(neighbors, reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

    for node in nodes:
        if node.node_id not in visited:
            order.append(node.node_id)

    return order


def build_graph_payload(payload: dict[str, Any], arrows_only: bool = False) -> dict[str, Any]:
    raw_nodes = payload.get("nodes", [])
    
    # Separate connectors and flow nodes
    all_geoms: list[NodeGeom] = []
    flow_geoms: list[NodeGeom] = []
    connector_geoms: list[NodeGeom] = []
    node_records: list[dict[str, Any]] = []

    for index, node in enumerate(raw_nodes, start=1):
        geom = _node_geom(node, fallback_index=index)
        if geom is None:
            continue
        
        all_geoms.append(geom)
        
        if geom.class_name in ARROW_CLASSES:
            connector_geoms.append(geom)
        elif geom.class_name in FLOW_CLASSES:
            flow_geoms.append(geom)
            node_records.append(_node_record(node, geom, index=index))

    # Build edges using hybrid approach
    edges = _build_edges_hybrid(flow_geoms, connector_geoms, arrows_only=arrows_only)
    execution_order = _topological_execution_order(flow_geoms, edges)
    
    # Statistics
    arrow_matched_count = sum(1 for e in edges if e.get("confidence") == "arrow_matched")
    geometric_count = sum(1 for e in edges if e.get("confidence") == "geometric_fallback")

    payload["graph"] = {
        "version": "3",  # Connector-first hybrid approach
        "node_count": len(node_records),
        "edge_count": len(edges),
        "arrow_count": len(connector_geoms),
        "edge_statistics": {
            "arrow_matched": arrow_matched_count,
            "geometric_fallback": geometric_count,
        },
        "nodes": node_records,
        "edges": edges,
        "entry_nodes": [node.node_id for node in flow_geoms if node.class_name == "start"],
        "execution_order": execution_order,
    }
    return payload


def _resolve_output_path(input_path: Path, output_arg: str, in_place: bool) -> Path:
    if in_place:
        return input_path

    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        return out

    out.mkdir(parents=True, exist_ok=True)
    name = input_path.name
    if name.lower().endswith(".post.ocr.json"):
        name = name[:-13] + ".graph.json"
    elif name.lower().endswith(".ocr.json"):
        name = name[:-9] + ".graph.json"
    else:
        name = input_path.stem + ".graph.json"
    return out / name


def process_one_file(input_path: Path, output_path: Path, arrows_only: bool = False) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    payload = build_graph_payload(payload, arrows_only=arrows_only)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    graph = payload.get("graph", {})
    stats = graph.get("edge_statistics", {})
    
    print(f"Processed: {input_path}")
    print(f"Saved: {output_path}")
    print(
        f"Graph: nodes={graph.get('node_count', 0)}, edges={graph.get('edge_count', 0)}, "
        f"arrows={graph.get('arrow_count', 0)}, entry={len(graph.get('entry_nodes', []))}"
    )
    print(f"Edges: arrow_matched={stats.get('arrow_matched', 0)}, geometric_fallback={stats.get('geometric_fallback', 0)}")


def main() -> None:
    args = parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    if src.is_file():
        output_path = _resolve_output_path(src, args.output, args.in_place)
        process_one_file(src, output_path, arrows_only=args.arrows_only)
        return

    files = sorted(src.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in: {src}")

    for file_path in files:
        output_path = _resolve_output_path(file_path, args.output, args.in_place)
        process_one_file(file_path, output_path, arrows_only=args.arrows_only)


if __name__ == "__main__":
    main()