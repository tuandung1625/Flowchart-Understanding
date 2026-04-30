"""Convert flowchart SVG annotations into YOLOv8 segmentation labels.

Pipeline:
1. Parse SVG geometry and optional .flowchart node mapping.
2. Export YOLOv8 segmentation .txt files.
3. Export Labelme-compatible JSON files for manual correction.
4. Optionally render overlay visualizations.

Example:
  .venv\Scripts\python.exe YOLO\convert_svg_to_yolo.py \
    --input DATASET\Train\svg \
    --flowcharts DATASET\Train\flowchart \
    --images DATASET\Train\images \
    --output DATASET\Train\labels \
    --labelme-output DATASET\Train\labelme \
    --visualize runs\svg_preview
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from svgpathtools import parse_path


CLASS_MAP = {
    "start": 0,
    "end": 1,
    "inputoutput": 2,
    "operation": 3,
    "subroutine": 4,
    "condition": 5,
    "arrow": 6,
    "arrow_head": 7,
}

CLASS_NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
NODE_CLASS_NAMES = ["start", "end", "inputoutput", "operation", "subroutine", "condition"]
ARROW_CLASS_NAMES = ["arrow", "arrow_head"]
DEFAULT_ARROW_THICKNESS_PX = 4.0


@dataclass
class Annotation:
    class_name: str
    polygon_px: list[tuple[float, float]]

    @property
    def class_id(self) -> int:
        return CLASS_MAP[self.class_name]

    @property
    def bbox_px(self) -> tuple[float, float, float, float]:
        xs = [pt[0] for pt in self.polygon_px]
        ys = [pt[1] for pt in self.polygon_px]
        return min(xs), min(ys), max(xs), max(ys)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def parse_flowchart(path: Path) -> dict[str, str]:
    node_types: dict[str, str] = {}
    if not path.exists():
        return node_types

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^(\w+)=>(\w+):", line.strip())
            if match:
                node_types[match.group(1)] = match.group(2)
    return node_types


def parse_style_attr(style_str: str) -> dict[str, str]:
    style: dict[str, str] = {}
    if not style_str:
        return style
    for item in style_str.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        style[key.strip()] = value.strip()
    return style


def svg_attr(elem: ET.Element, name: str, default: str = "") -> str:
    value = elem.get(name)
    if value not in (None, ""):
        return value
    style = parse_style_attr(elem.get("style", ""))
    return style.get(name, default)


def parse_transform(transform_str: str) -> np.ndarray:
    matrix = np.eye(3, dtype=float)
    if not transform_str:
        return matrix

    for match in re.finditer(r"([a-zA-Z]+)\(([^)]*)\)", transform_str):
        name = match.group(1).strip()
        raw_values = [token for token in re.split(r"[\s,]+", match.group(2).strip()) if token]
        values = [float(token) for token in raw_values]

        op = np.eye(3, dtype=float)
        if name == "matrix" and len(values) == 6:
            a, b, c, d, tx, ty = values
            op = np.array([[a, c, tx], [b, d, ty], [0.0, 0.0, 1.0]], dtype=float)
        elif name == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            op = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
        elif name == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            op = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                op = (
                    np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=float)
                    @ rot
                    @ np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=float)
                )
            else:
                op = rot
        elif name == "skewX" and values:
            angle = math.radians(values[0])
            op = np.array([[1.0, math.tan(angle), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        elif name == "skewY" and values:
            angle = math.radians(values[0])
            op = np.array([[1.0, 0.0, 0.0], [math.tan(angle), 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

        matrix = matrix @ op
    return matrix


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    result = matrix @ np.array([x, y, 1.0], dtype=float)
    return float(result[0]), float(result[1])


def transform_points(matrix: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [transform_point(matrix, x, y) for x, y in points]


def viewbox_and_size(root: ET.Element) -> tuple[float, float, float, float]:
    viewbox = root.get("viewBox", "").split()
    if len(viewbox) >= 4:
        vb_x, vb_y, vb_w, vb_h = map(float, viewbox[:4])
    else:
        vb_x = 0.0
        vb_y = 0.0
        vb_w = float(root.get("width", 1) or 1)
        vb_h = float(root.get("height", 1) or 1)
    return vb_x, vb_y, vb_w, vb_h


def canvas_size_from_svg(root: ET.Element) -> tuple[int, int]:
    width = root.get("width", "")
    height = root.get("height", "")

    def to_float(value: str, fallback: float) -> float:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value or "")
        return float(match.group(0)) if match else fallback

    vb_x, vb_y, vb_w, vb_h = viewbox_and_size(root)
    return int(round(to_float(width, vb_w))), int(round(to_float(height, vb_h)))


def norm_xy_to_px(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    px = int(round(clamp01(x) * max(0, width - 1)))
    py = int(round(clamp01(y) * max(0, height - 1)))
    return px, py


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def path_bbox(d: str) -> tuple[float, float, float, float] | None:
    try:
        path = parse_path(d)
        bbox = path.bbox()
        if bbox is None or len(bbox) != 4:
            return None
        x_min, x_max, y_min, y_max = bbox
        return x_min, y_min, x_max - x_min, y_max - y_min
    except Exception:
        return None


def path_to_vertices(d: str, num_samples: int = 24) -> list[tuple[float, float]] | None:
    try:
        path = parse_path(d)
        if not path:
            return None

        if all(seg.__class__.__name__ == "Line" for seg in path):
            vertices = [(path[0].start.real, path[0].start.imag)]
            for seg in path:
                end_pt = (seg.end.real, seg.end.imag)
                if end_pt != vertices[-1]:
                    vertices.append(end_pt)
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices.pop()
            return vertices

        vertices: list[tuple[float, float]] = []
        seg_samples = max(6, num_samples // max(1, len(path)))
        for seg in path:
            if not vertices:
                vertices.append((seg.start.real, seg.start.imag))
            for idx in range(1, seg_samples + 1):
                t = idx / seg_samples
                point = seg.point(t)
                pt = (point.real, point.imag)
                if pt != vertices[-1]:
                    vertices.append(pt)

        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()
        return vertices if vertices else None
    except Exception:
        return None


def polygon_attr_to_vertices(points_str: str) -> list[tuple[float, float]] | None:
    if not points_str:
        return None
    tokens = [token for token in re.split(r"[\s,]+", points_str.strip()) if token]
    if len(tokens) < 6 or len(tokens) % 2 != 0:
        return None
    vertices = [(float(tokens[idx]), float(tokens[idx + 1])) for idx in range(0, len(tokens), 2)]
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    return vertices if len(vertices) >= 3 else None


def rect_to_vertices(
    x: float,
    y: float,
    w: float,
    h: float,
    rx: float = 0.0,
    ry: float = 0.0,
    arc_samples: int = 6,
) -> list[tuple[float, float]]:
    rx = max(0.0, min(rx, w / 2))
    ry = max(0.0, min(ry, h / 2))

    if rx == 0.0 and ry > 0.0:
        rx = min(ry, w / 2)
    if ry == 0.0 and rx > 0.0:
        ry = min(rx, h / 2)

    if rx == 0.0 and ry == 0.0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def arc_points(
        cx: float,
        cy: float,
        arx: float,
        ary: float,
        start_angle: float,
        end_angle: float,
        steps: int,
    ) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for idx in range(1, steps + 1):
            t = idx / steps
            ang = start_angle + (end_angle - start_angle) * t
            pts.append((cx + arx * math.cos(ang), cy + ary * math.sin(ang)))
        return pts

    n = max(3, arc_samples)
    vertices = [(x + rx, y), (x + w - rx, y)]
    vertices.extend(arc_points(x + w - rx, y + ry, rx, ry, -math.pi / 2, 0.0, n))
    vertices.append((x + w, y + h - ry))
    vertices.extend(arc_points(x + w - rx, y + h - ry, rx, ry, 0.0, math.pi / 2, n))
    vertices.append((x + rx, y + h))
    vertices.extend(arc_points(x + rx, y + h - ry, rx, ry, math.pi / 2, math.pi, n))
    vertices.append((x, y + ry))
    vertices.extend(arc_points(x + rx, y + ry, rx, ry, math.pi, 3 * math.pi / 2, n))

    cleaned: list[tuple[float, float]] = []
    for point in vertices:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def stroke_polyline_to_vertices(points: list[tuple[float, float]], thickness: float) -> list[tuple[float, float]] | None:
    if len(points) < 2:
        return None

    clean_points: list[tuple[float, float]] = [points[0]]
    for point in points[1:]:
        if point != clean_points[-1]:
            clean_points.append(point)

    if len(clean_points) < 2:
        return None

    half = thickness / 2.0

    def unit(dx: float, dy: float) -> tuple[float, float]:
        length = math.hypot(dx, dy)
        if length == 0.0:
            return 0.0, 0.0
        return dx / length, dy / length

    def normal(dx: float, dy: float) -> tuple[float, float]:
        ux, uy = unit(dx, dy)
        return -uy, ux

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    count = len(clean_points)

    for idx, (x, y) in enumerate(clean_points):
        if idx == 0:
            nx, ny = normal(clean_points[1][0] - x, clean_points[1][1] - y)
        elif idx == count - 1:
            nx, ny = normal(x - clean_points[idx - 1][0], y - clean_points[idx - 1][1])
        else:
            n1x, n1y = normal(x - clean_points[idx - 1][0], y - clean_points[idx - 1][1])
            n2x, n2y = normal(clean_points[idx + 1][0] - x, clean_points[idx + 1][1] - y)
            nx, ny = n1x + n2x, n1y + n2y
            norm = math.hypot(nx, ny)
            if norm == 0.0:
                nx, ny = n2x, n2y
            else:
                nx, ny = nx / norm, ny / norm

        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))

    polygon = left + right[::-1]
    cleaned: list[tuple[float, float]] = []
    for point in polygon:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else None


def arrow_head_triangle(end_point: tuple[float, float], prev_point: tuple[float, float], size: float) -> list[tuple[float, float]] | None:
    dx = end_point[0] - prev_point[0]
    dy = end_point[1] - prev_point[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None

    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base = (end_point[0] - ux * size, end_point[1] - uy * size)
    left = (base[0] + px * size * 0.45, base[1] + py * size * 0.45)
    right = (base[0] - px * size * 0.45, base[1] - py * size * 0.45)
    return [left, end_point, right]


def is_probable_yolo_bbox(xc: float, yc: float, w: float, h: float) -> bool:
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        return False
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return -0.05 <= x1 <= 1.05 and -0.05 <= y1 <= 1.05 and -0.05 <= x2 <= 1.05 and -0.05 <= y2 <= 1.05


def polygons_to_yolo_line(class_id: int, polygon_px: list[tuple[float, float]], width: int, height: int) -> str:
    points = []
    for x, y in polygon_px:
        points.append(f"{clamp01(x / max(1, width - 1)):.6f}")
        points.append(f"{clamp01(y / max(1, height - 1)):.6f}")
    return f"{class_id} " + " ".join(points)


def svg_centerline_points(elem: ET.Element, tag: str) -> list[tuple[float, float]] | None:
    if tag == "path":
        return path_to_vertices(elem.get("d", ""), num_samples=32)
    if tag == "line":
        return [
            (float(elem.get("x1", 0) or 0), float(elem.get("y1", 0) or 0)),
            (float(elem.get("x2", 0) or 0), float(elem.get("y2", 0) or 0)),
        ]
    if tag == "polyline":
        return polygon_attr_to_vertices(elem.get("points", ""))
    return None


def close_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    if len(points) > 1 and points[0] == points[-1]:
        return points
    return points + [points[0]]


def dedupe_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(points) - 1):
        total += point_distance(points[idx], points[idx + 1])
    return total


def is_closed_polygon(points: list[tuple[float, float]], tolerance: float = 2.0) -> bool:
    if len(points) < 3:
        return False
    return point_distance(points[0], points[-1]) <= tolerance


def close_polygon_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    if is_closed_polygon(points):
        return points
    return points + [points[0]]


def is_compact_closed_shape(points: list[tuple[float, float]], canvas_width: int, canvas_height: int) -> bool:
    if len(points) < 3:
        return False
    bbox = polygon_bbox(points)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return False
    area_ratio = polygon_area(points) / max(1.0, float(canvas_width * canvas_height))
    bbox_ratio = (width * height) / max(1.0, float(canvas_width * canvas_height))
    # Nodes are closed and compact; this prevents long paths or tiny noise from
    # being treated as node shapes.
    return area_ratio >= 1e-4 and bbox_ratio <= 0.25 and max(width, height) <= max(canvas_width, canvas_height) * 0.9


def is_arrow_head_candidate(points: list[tuple[float, float]], canvas_width: int, canvas_height: int) -> bool:
    if len(points) < 3 or not is_closed_polygon(points):
        return False
    bbox = polygon_bbox(points)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area_ratio = polygon_area(points) / max(1.0, float(canvas_width * canvas_height))
    bbox_ratio = (width * height) / max(1.0, float(canvas_width * canvas_height))
    return area_ratio <= 0.02 and bbox_ratio <= 0.02 and max(width, height) <= max(canvas_width, canvas_height) * 0.25


def edge_endpoint_pair(points: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(points) < 2:
        return None
    return points[0], points[-1]


def nearest_endpoint_distance(points: list[tuple[float, float]], endpoints: list[tuple[float, float]]) -> float:
    if not points or not endpoints:
        return float("inf")
    best = float("inf")
    for pt in points:
        for endpoint in endpoints:
            best = min(best, point_distance(pt, endpoint))
    return best


def is_open_edge_path(
    elem: ET.Element,
    tag: str,
    polygon_points: list[tuple[float, float]],
    canvas_width: int,
    canvas_height: int,
) -> bool:
    if tag not in {"path", "line", "polyline"}:
        return False

    fill = svg_attr(elem, "fill", "none")
    stroke = svg_attr(elem, "stroke", "")
    if stroke == "none":
        return False
    if fill != "none":
        return False
    if len(polygon_points) < 2 or is_closed_polygon(polygon_points):
        return False

    length = path_length(polygon_points)
    bbox = polygon_bbox(polygon_points)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    # Long, open, thin geometry is treated as an edge.
    return length >= min(canvas_width, canvas_height) * 0.08 and max(width, height) >= min(canvas_width, canvas_height) * 0.05


def extract_node_annotations(
    all_elements: list[tuple[ET.Element, str, np.ndarray]],
    node_types: dict[str, str],
    viewbox_to_image: np.ndarray,
    arrow_thickness_svg: float,
    canvas_width: int,
    canvas_height: int,
) -> tuple[list[Annotation], list[tuple[float, float, float, float]], set[int]]:
    node_annotations: list[Annotation] = []
    node_boxes: list[tuple[float, float, float, float]] = []
    consumed_ids: set[int] = set()

    for idx, (elem, tag, inherited_matrix) in enumerate(all_elements):
        elem_id = elem.get("id", "")
        if not elem_id or elem_id not in node_types:
            continue

        class_name = node_types[elem_id]
        if class_name not in NODE_CLASS_NAMES:
            continue

        combined_matrix = viewbox_to_image @ inherited_matrix
        vertices = element_vertices(elem, tag, combined_matrix, arrow_thickness_svg, forced_class=class_name)
        if not vertices:
            continue

        vertices = close_polygon_points(dedupe_polygon(vertices))
        if len(vertices) < 4:
            continue
        if not is_closed_polygon(vertices):
            continue
        if not is_compact_closed_shape(vertices, canvas_width, canvas_height):
            continue

        node_annotations.append(Annotation(class_name=class_name, polygon_px=vertices))
        bbox = polygon_bbox(vertices)
        if bbox is not None:
            node_boxes.append(bbox)
        consumed_ids.add(idx)

    return node_annotations, node_boxes, consumed_ids


def mask_node_regions_from_path(
    polygon_points: list[tuple[float, float]],
    node_boxes: list[tuple[float, float, float, float]],
) -> bool:
    if not polygon_points:
        return False
    bbox = polygon_bbox(polygon_points)
    if bbox is None:
        return False
    return any(bbox_iou(bbox, node_bbox) > 0.35 for node_bbox in node_boxes)


def normalize_vertices(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for x, y in points:
        nx = clamp01(x / max(1, width - 1))
        ny = clamp01(y / max(1, height - 1))
        normalized.append((nx, ny))
    return normalized


def sample_svg_elements(root: ET.Element) -> list[tuple[ET.Element, str, np.ndarray]]:
    elements: list[tuple[ET.Element, str, np.ndarray]] = []

    def walk(elem: ET.Element, inherited_matrix: np.ndarray) -> None:
        matrix = inherited_matrix @ parse_transform(elem.get("transform", ""))
        tag = elem.tag.split("}")[-1]
        elements.append((elem, tag, matrix))
        for child in list(elem):
            walk(child, matrix)

    walk(root, np.eye(3, dtype=float))
    return elements


def element_vertices(
    elem: ET.Element,
    tag: str,
    matrix: np.ndarray,
    arrow_thickness_svg: float,
    forced_class: str | None = None,
) -> list[tuple[float, float]] | None:
    """Extract vertices from SVG element and apply transformation matrix.
    
    Returns vertices in SVG coordinate space (already transformed by matrix).
    """
    fill = svg_attr(elem, "fill", "none")
    stroke = svg_attr(elem, "stroke", "")

    vertices: list[tuple[float, float]] | None = None

    if tag == "rect":
        x = float(elem.get("x", 0) or 0)
        y = float(elem.get("y", 0) or 0)
        w = float(elem.get("width", 0) or 0)
        h = float(elem.get("height", 0) or 0)
        rx = float(elem.get("rx", 0) or 0)
        ry = float(elem.get("ry", 0) or 0)
        # Keep rounded corners for nodes that actually use rx/ry (e.g. start/end).
        # Only fall back to exact corners for square rectangles to avoid any
        # unnecessary arc expansion.
        if forced_class in NODE_CLASS_NAMES and (rx > 0.0 or ry > 0.0):
            vertices = rect_to_vertices(x, y, w, h, rx=rx, ry=ry)
        elif forced_class in NODE_CLASS_NAMES:
            vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            vertices = rect_to_vertices(x, y, w, h, rx=rx, ry=ry)
    elif tag in {"polygon", "polyline"}:
        points = polygon_attr_to_vertices(elem.get("points", ""))
        if points:
            vertices = points
    elif tag == "line":
        x1 = float(elem.get("x1", 0) or 0)
        y1 = float(elem.get("y1", 0) or 0)
        x2 = float(elem.get("x2", 0) or 0)
        y2 = float(elem.get("y2", 0) or 0)
        vertices = [(x1, y1), (x2, y2)]
    elif tag == "path":
        d = elem.get("d", "")
        if not d:
            return None
        path_vertices = path_to_vertices(d, num_samples=32)
        if path_vertices:
            vertices = path_vertices

    if not vertices:
        return None

    # Apply element transformation matrix
    transformed = transform_points(matrix, vertices)

    if forced_class == "arrow":
        if len(transformed) < 2:
            return None
        thick = stroke_polyline_to_vertices(transformed, thickness=arrow_thickness_svg)
        return dedupe_polygon(thick) if thick else None

    if forced_class == "arrow_head":
        if len(transformed) >= 3:
            return dedupe_polygon(transformed)
        if len(transformed) == 2:
            triangle = arrow_head_triangle(transformed[1], transformed[0], max(arrow_thickness_svg * 2.0, 6.0))
            return triangle
        return None

    if tag == "path" and fill != "none":
        if len(transformed) >= 3:
            return dedupe_polygon(transformed)
        return None

    if tag in {"line", "polyline"} and stroke != "none":
        if len(transformed) >= 2:
            thick = stroke_polyline_to_vertices(transformed, thickness=arrow_thickness_svg)
            return dedupe_polygon(thick) if thick else None

    if tag in {"polygon", "rect"}:
        return dedupe_polygon(transformed)

    return None


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    if not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def extract_annotations(
    svg_path: Path,
    flowchart_path: Path | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[Annotation], int, int]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    svg_width, svg_height = canvas_size_from_svg(root)
    canvas_width = image_size[0] if image_size else svg_width
    canvas_height = image_size[1] if image_size else svg_height

    vb_x, vb_y, vb_w, vb_h = viewbox_and_size(root)
    if vb_w <= 0 or vb_h <= 0:
        vb_w = float(canvas_width or 1)
        vb_h = float(canvas_height or 1)

    scale_x = canvas_width / vb_w if vb_w else 1.0
    scale_y = canvas_height / vb_h if vb_h else 1.0
    scale = min(scale_x, scale_y)
    offset_x = (canvas_width - vb_w * scale) / 2.0 - vb_x * scale
    offset_y = (canvas_height - vb_h * scale) / 2.0 - vb_y * scale
    viewbox_to_image = np.array([[scale, 0.0, offset_x], [0.0, scale, offset_y], [0.0, 0.0, 1.0]], dtype=float)

    node_types = parse_flowchart(flowchart_path) if flowchart_path else {}
    all_elements = sample_svg_elements(root)
    arrow_thickness_svg = DEFAULT_ARROW_THICKNESS_PX / max(1.0, min(scale_x, scale_y))

    annotations: list[Annotation] = []
    node_annotations, node_boxes, consumed_ids = extract_node_annotations(
        all_elements=all_elements,
        node_types=node_types,
        viewbox_to_image=viewbox_to_image,
        arrow_thickness_svg=arrow_thickness_svg,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    annotations.extend(node_annotations)

    edge_endpoints: list[tuple[float, float]] = []
    edge_centerlines: list[list[tuple[float, float]]] = []

    # Step 2: remove node region and detect EDGE (open long paths).
    for idx, (elem, tag, inherited_matrix) in enumerate(all_elements):
        if idx in consumed_ids:
            continue

        elem_id = elem.get("id", "")
        if elem_id and elem_id in node_types and node_types[elem_id] in NODE_CLASS_NAMES:
            continue

        combined_matrix = viewbox_to_image @ inherited_matrix
        centerline_points = svg_centerline_points(elem, tag)
        if not centerline_points or len(centerline_points) < 2:
            continue

        centerline = transform_points(combined_matrix, centerline_points)
        if not is_open_edge_path(elem, tag, centerline, canvas_width, canvas_height):
            continue

        thick_polygon = stroke_polyline_to_vertices(centerline, thickness=arrow_thickness_svg)
        if not thick_polygon:
            continue

        polygon = dedupe_polygon(thick_polygon)
        if len(polygon) < 3:
            continue

        # If this geometry overlaps a stored node box, keep it out of the edge set.
        if mask_node_regions_from_path(polygon, node_boxes):
            if polygon_area(polygon) < (canvas_width * canvas_height) * 0.01:
                continue

        annotations.append(Annotation(class_name="arrow", polygon_px=polygon))
        edge_centerlines.append(centerline)
        edge_endpoints.append(centerline[0])
        edge_endpoints.append(centerline[-1])
        consumed_ids.add(idx)

    # Step 3: detect ARROW HEAD as small closed polygon near edge endpoints.
    for idx, (elem, tag, inherited_matrix) in enumerate(all_elements):
        if idx in consumed_ids:
            continue

        elem_id = elem.get("id", "")
        if elem_id and elem_id in node_types and node_types[elem_id] in NODE_CLASS_NAMES:
            continue

        fill = svg_attr(elem, "fill", "none")
        if tag not in {"polygon", "path"}:
            continue

        combined_matrix = viewbox_to_image @ inherited_matrix
        vertices = element_vertices(elem, tag, combined_matrix, arrow_thickness_svg)
        if not vertices:
            continue

        vertices = dedupe_polygon(vertices)
        if len(vertices) < 3 or not is_closed_polygon(vertices):
            continue
        if not is_arrow_head_candidate(vertices, canvas_width, canvas_height):
            continue

        if edge_endpoints:
            proximity = nearest_endpoint_distance(vertices, edge_endpoints)
            if proximity > max(arrow_thickness_svg * 12.0, min(canvas_width, canvas_height) * 0.08):
                continue

        annotations.append(Annotation(class_name="arrow_head", polygon_px=vertices))
        consumed_ids.add(idx)

    # Step 4: if an edge does not have a separate head, approximate one at the endpoint.
    detected_heads = [ann for ann in annotations if ann.class_name == "arrow_head"]
    detected_head_boxes = [polygon_bbox(ann.polygon_px) for ann in detected_heads if polygon_bbox(ann.polygon_px) is not None]

    for centerline in edge_centerlines:
        if len(centerline) < 2:
            continue
        endpoint = centerline[-1]
        already_covered = False
        for head_box in detected_head_boxes:
            if point_in_bbox(endpoint, head_box):
                already_covered = True
                break
        if already_covered:
            continue

        head = arrow_head_triangle(centerline[-1], centerline[-2], max(arrow_thickness_svg * 2.0, 6.0))
        if not head:
            continue
        annotations.append(Annotation(class_name="arrow_head", polygon_px=dedupe_polygon(head)))

    annotations = sorted(annotations, key=lambda ann: CLASS_MAP[ann.class_name])
    return annotations, canvas_width, canvas_height


def export_yolo_labels(output_path: Path, annotations: list[Annotation], width: int, height: int) -> None:
    lines: list[str] = []
    for ann in annotations:
        polygon_norm = normalize_vertices(ann.polygon_px, width, height)
        if len(polygon_norm) < 3:
            continue
        points = []
        for x, y in polygon_norm:
            points.append(f"{x:.6f}")
            points.append(f"{y:.6f}")
        lines.append(f"{ann.class_id} " + " ".join(points))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def export_labelme_json(
    output_path: Path,
    image_path: Path,
    annotations: list[Annotation],
    width: int,
    height: int,
) -> None:
    shapes = []
    for ann in annotations:
        shapes.append(
            {
                "label": ann.class_name,
                "points": [[float(x), float(y)] for x, y in close_polygon(dedupe_polygon(ann.polygon_px))],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )

    payload = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]
    return palette[class_id % len(palette)]


def draw_overlay(image_path: Path, annotations: list[Annotation], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for ann in annotations:
        color = color_for_class(ann.class_id)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in ann.polygon_px], dtype=np.int32)
        if len(pts) >= 2:
            # Always draw polygon outline
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            # Fill only for arrow classes (6,7). For node classes (0-5) we
            # intentionally do NOT fill to avoid the large dark segmentation
            # patch that visually enlarges the node.
            if len(pts) >= 3 and ann.class_id not in range(0, 6):
                cv2.fillPoly(image, [pts], color=(color[0] // 6, color[1] // 6, color[2] // 6))

        x1, y1, x2, y2 = [int(round(v)) for v in ann.bbox_px]
        # Draw bbox rectangle only for non-node classes (keep node visuals clean)
        if ann.class_id not in range(0, 6):
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            image,
            f"{ann.class_name} ({ann.class_id})",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def iter_svg_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.rglob("*.svg"))


def process_file(
    svg_path: Path,
    flowcharts_dir: Path | None,
    images_dir: Path | None,
    output_dir: Path,
    labelme_dir: Path | None,
    visualize_dir: Path | None,
) -> bool:
    image_path = images_dir / f"{svg_path.stem}.png" if images_dir else None
    flowchart_path = flowcharts_dir / f"{svg_path.stem}.flowchart" if flowcharts_dir else None

    image_size = read_image_size(image_path) if image_path else None
    annotations, width, height = extract_annotations(svg_path, flowchart_path=flowchart_path, image_size=image_size)
    if not annotations:
        return False

    yolo_path = output_dir / f"{svg_path.stem}.txt"
    export_yolo_labels(yolo_path, annotations, width, height)

    if labelme_dir is not None:
        json_path = labelme_dir / f"{svg_path.stem}.json"
        export_labelme_json(json_path, image_path if image_path else svg_path.with_suffix(".png"), annotations, width, height)

    if visualize_dir is not None and image_path is not None:
        preview_path = visualize_dir / f"{svg_path.stem}.png"
        draw_overlay(image_path, annotations, preview_path)

    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SVG flowchart annotations to YOLOv8 segmentation labels.")
    parser.add_argument("--input", type=Path, required=True, help="Folder containing SVG files")
    parser.add_argument("--output", type=Path, required=True, help="Folder for YOLO .txt labels")
    parser.add_argument("--flowcharts", type=Path, default=None, help="Optional folder with .flowchart files")
    parser.add_argument("--images", type=Path, default=None, help="Optional folder with PNG images")
    parser.add_argument("--labelme-output", type=Path, default=None, help="Optional folder for Labelme JSON output")
    parser.add_argument("--visualize", type=Path, default=None, help="Optional folder for overlay preview images")
    parser.add_argument("--max-files", type=int, default=0, help="Limit the number of SVG files processed (0 = all)")
    parser.add_argument("--arrow-thickness", type=float, default=DEFAULT_ARROW_THICKNESS_PX, help="Approximate arrow thickness in pixels")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    global DEFAULT_ARROW_THICKNESS_PX
    DEFAULT_ARROW_THICKNESS_PX = args.arrow_thickness

    svg_files = list(iter_svg_files(args.input))
    if args.max_files > 0:
        svg_files = svg_files[: args.max_files]

    args.output.mkdir(parents=True, exist_ok=True)
    if args.labelme_output is not None:
        args.labelme_output.mkdir(parents=True, exist_ok=True)
    if args.visualize is not None:
        args.visualize.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for svg_path in svg_files:
        ok = process_file(
            svg_path=svg_path,
            flowcharts_dir=args.flowcharts,
            images_dir=args.images,
            output_dir=args.output,
            labelme_dir=args.labelme_output,
            visualize_dir=args.visualize,
        )
        if ok:
            processed += 1
        else:
            skipped += 1

    print("Conversion complete")
    print(f"  Input folder   : {args.input}")
    print(f"  Output folder  : {args.output}")
    print(f"  Labelme folder : {args.labelme_output}")
    print(f"  Visualize folder: {args.visualize}")
    print(f"  Processed      : {processed}")
    print(f"  Skipped        : {skipped}")


if __name__ == "__main__":
    main()