"""Convert flowchart SVG annotations into YOLOv8 segmentation labels.

New approach:
1. Use marker-end attribute to detect arrows (more reliable)
2. Parse marker definitions to get arrow head shape
3. Sample path to create thin arrow shaft polygon
4. Export separate annotations for shaft and head

Example:
  python convert_svg_to_yolo_v2.py \
    --input DATASET/Train/svg \
    --flowcharts DATASET/Train/flowchart \
    --images DATASET/Train/images \
    --output DATASET/Train/labels \
    --labelme-output DATASET/Train/labelme \
    --visualize runs/svg_preview
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

# Arrow shaft thickness: minimal to represent connectivity, not size
ARROW_SHAFT_THICKNESS_PX = 8.0
# Arrow head: typical marker size
ARROW_HEAD_SIZE_PX = 20.0


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_flowchart(path: Path) -> dict[str, str]:
    """Parse .flowchart file to get node ID -> type mapping."""
    node_types: dict[str, str] = {}
    if not path or not path.exists():
        return node_types

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^(\w+)=>(\w+):", line.strip())
            if match:
                node_types[match.group(1)] = match.group(2)
    return node_types


def parse_style_attr(style_str: str) -> dict[str, str]:
    """Parse inline style attribute into dict."""
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
    """Get attribute value from element or its style."""
    value = elem.get(name)
    if value not in (None, ""):
        return value
    style = parse_style_attr(elem.get("style", ""))
    return style.get(name, default)


def parse_transform(transform_str: str) -> np.ndarray:
    """Parse SVG transform attribute into 3x3 matrix."""
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

        matrix = matrix @ op
    return matrix


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply transformation matrix to a point."""
    result = matrix @ np.array([x, y, 1.0], dtype=float)
    return float(result[0]), float(result[1])


def transform_points(matrix: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply transformation matrix to list of points."""
    return [transform_point(matrix, x, y) for x, y in points]


def viewbox_and_size(root: ET.Element) -> tuple[float, float, float, float]:
    """Extract viewBox parameters from SVG root."""
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
    """Get canvas size from SVG width/height attributes."""
    width = root.get("width", "")
    height = root.get("height", "")

    def to_float(value: str, fallback: float) -> float:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value or "")
        return float(match.group(0)) if match else fallback

    vb_x, vb_y, vb_w, vb_h = viewbox_and_size(root)
    return int(round(to_float(width, vb_w))), int(round(to_float(height, vb_h)))


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def path_to_vertices(d: str, num_samples: int = 24) -> list[tuple[float, float]] | None:
    """Convert SVG path d-attribute to sampled vertices."""
    try:
        path = parse_path(d)
        if not path:
            return None

        # For pure line paths, use endpoints only
        if all(seg.__class__.__name__ == "Line" for seg in path):
            vertices = [(path[0].start.real, path[0].start.imag)]
            for seg in path:
                end_pt = (seg.end.real, seg.end.imag)
                if end_pt != vertices[-1]:
                    vertices.append(end_pt)
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices.pop()
            return vertices

        # For curves, sample along path
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
    """Parse polygon/polyline points attribute."""
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
    """Convert rect element to polygon vertices with optional rounded corners."""
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


def stroke_polyline_thin(
    points: list[tuple[float, float]], 
    thickness: float
) -> list[tuple[float, float]] | None:
    """Create thin stroked polygon from polyline centerline."""
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


def arrow_head_triangle(
    end_point: tuple[float, float], 
    direction: tuple[float, float], 
    size: float
) -> list[tuple[float, float]] | None:
    """Create arrow head triangle at endpoint."""
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None

    # Unit vector in arrow direction
    ux, uy = dx / length, dy / length
    # Perpendicular vector
    px, py = -uy, ux
    
    # Triangle vertices: tip + two base points
    tip = (end_point[0], end_point[1] + size/2)
    base_center = (end_point[0] - ux * size/1.2, end_point[1] - uy * size/1.2)
    left = (base_center[0] + px * size * 0.8, base_center[1] + py * size * 0.5)
    right = (base_center[0] - px * size * 0.8, base_center[1] - py * size * 0.5)
    
    return [tip, left, right]


def dedupe_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove consecutive duplicate points."""
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def close_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Close polygon by adding first point at end if not already closed."""
    if not points:
        return points
    if len(points) > 1 and points[0] == points[-1]:
        return points
    return points + [points[0]]


def normalize_vertices(
    points: list[tuple[float, float]], 
    width: int, 
    height: int
) -> list[tuple[float, float]]:
    """Normalize vertices to [0, 1] range."""
    normalized: list[tuple[float, float]] = []
    for x, y in points:
        nx = clamp01(x / max(1, width - 1))
        ny = clamp01(y / max(1, height - 1))
        normalized.append((nx, ny))
    return normalized


def parse_marker_reference(marker_url: str) -> str | None:
    """Extract marker ID from url(#marker-id) format."""
    match = re.match(r"url\(#(.+)\)", marker_url)
    return match.group(1) if match else None


def find_marker_def(root: ET.Element, marker_id: str) -> ET.Element | None:
    """Find marker definition in SVG defs."""
    # Search in all defs sections
    for defs in root.iter("{http://www.w3.org/2000/svg}defs"):
        for marker in defs.iter("{http://www.w3.org/2000/svg}marker"):
            if marker.get("id") == marker_id:
                return marker
    # Also check without namespace
    for defs in root.iter("defs"):
        for marker in defs.iter("marker"):
            if marker.get("id") == marker_id:
                return marker
    return None


def extract_marker_path(marker: ET.Element) -> str | None:
    """Extract path d-attribute from marker definition."""
    # Marker contains <use> pointing to <path>
    for use in marker.iter():
        tag = use.tag.split("}")[-1]
        if tag == "use":
            href = use.get("{http://www.w3.org/1999/xlink}href") or use.get("href", "")
            if href.startswith("#"):
                path_id = href[1:]
                # Find the referenced path in defs
                root = marker
                while root.getparent() is not None:
                    root = root.getparent()
                for path_elem in root.iter():
                    if path_elem.get("id") == path_id:
                        return path_elem.get("d", "")
    
    # Or marker directly contains path
    for path in marker.iter():
        tag = path.tag.split("}")[-1]
        if tag == "path":
            return path.get("d", "")
    
    return None


def walk_elements(
    elem: ET.Element, 
    inherited_matrix: np.ndarray
) -> list[tuple[ET.Element, str, np.ndarray]]:
    """Recursively walk SVG tree and collect elements with transforms."""
    elements: list[tuple[ET.Element, str, np.ndarray]] = []
    matrix = inherited_matrix @ parse_transform(elem.get("transform", ""))
    tag = elem.tag.split("}")[-1]
    elements.append((elem, tag, matrix))
    
    for child in list(elem):
        elements.extend(walk_elements(child, matrix))
    
    return elements


def extract_annotations(
    svg_path: Path,
    flowchart_path: Path | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[Annotation], int, int]:
    """Extract YOLO segmentation annotations from SVG flowchart.
    
    New approach:
    1. Parse nodes using flowchart mapping or class attribute
    2. Detect arrows using marker-end attribute (more reliable than heuristics)
    3. Create thin shaft polygon + separate arrow head polygon
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get canvas dimensions
    svg_width, svg_height = canvas_size_from_svg(root)
    canvas_width = image_size[0] if image_size else svg_width
    canvas_height = image_size[1] if image_size else svg_height

    # Build viewBox to image coordinate transform
    vb_x, vb_y, vb_w, vb_h = viewbox_and_size(root)
    if vb_w <= 0 or vb_h <= 0:
        vb_w = float(canvas_width or 1)
        vb_h = float(canvas_height or 1)

    scale_x = canvas_width / vb_w if vb_w else 1.0
    scale_y = canvas_height / vb_h if vb_h else 1.0
    scale = min(scale_x, scale_y)
    offset_x = (canvas_width - vb_w * scale) / 2.0 - vb_x * scale
    offset_y = (canvas_height - vb_h * scale) / 2.0 - vb_y * scale
    viewbox_to_image = np.array(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y], [0.0, 0.0, 1.0]], 
        dtype=float
    )

    # Parse node type mapping from .flowchart file
    node_types = parse_flowchart(flowchart_path) if flowchart_path else {}
    
    # Walk all elements
    all_elements = walk_elements(root, np.eye(3, dtype=float))
    
    # *** FIXED: Arrow dimensions are now directly in pixel space ***
    arrow_shaft_thickness_px = ARROW_SHAFT_THICKNESS_PX
    arrow_head_size_px = ARROW_HEAD_SIZE_PX

    annotations: list[Annotation] = []

    # ====================
    # Step 1: Extract NODES
    # ====================
    for elem, tag, inherited_matrix in all_elements:
        elem_id = elem.get("id", "")
        
        # Check if this is a flowchart node
        class_name = None
        if elem_id and elem_id in node_types:
            class_name = node_types[elem_id]
        elif svg_attr(elem, "class", "") == "flowchart":
            # Fallback: detect node type from shape
            if tag == "rect":
                rx = float(elem.get("rx", 0) or 0)
                if rx > 0:
                    class_name = "start"  # or "end" - can't distinguish without .flowchart
                else:
                    class_name = "operation"
            elif tag == "path":
                # Diamond shape
                class_name = "condition"
        
        if class_name not in NODE_CLASS_NAMES:
            continue

        # Extract node polygon
        combined_matrix = viewbox_to_image @ inherited_matrix
        vertices = None
        
        if tag == "rect":
            x = float(elem.get("x", 0) or 0)
            y = float(elem.get("y", 0) or 0)
            w = float(elem.get("width", 0) or 0)
            h = float(elem.get("height", 0) or 0)
            rx = float(elem.get("rx", 0) or 0)
            ry = float(elem.get("ry", 0) or 0)
            vertices = rect_to_vertices(x, y, w, h, rx, ry)
        elif tag == "path":
            d = elem.get("d", "")
            vertices = path_to_vertices(d, num_samples=32)
        elif tag == "polygon":
            vertices = polygon_attr_to_vertices(elem.get("points", ""))
        
        if not vertices:
            continue
        
        vertices = transform_points(combined_matrix, vertices)
        vertices = dedupe_polygon(vertices)
        
        if len(vertices) >= 3:
            annotations.append(Annotation(class_name=class_name, polygon_px=vertices))

    # ====================
    # Step 2: Extract ARROWS using marker-end
    # ====================
    for elem, tag, inherited_matrix in all_elements:
        # Check if element has marker-end attribute (indicates arrow)
        marker_end = svg_attr(elem, "marker-end", "")
        if not marker_end or marker_end == "none":
            continue
        
        # Must be a path-like element
        if tag not in {"path", "line", "polyline"}:
            continue
        
        # Extract centerline path
        centerline = None
        if tag == "path":
            d = elem.get("d", "")
            centerline = path_to_vertices(d, num_samples=32)
        elif tag == "line":
            x1 = float(elem.get("x1", 0) or 0)
            y1 = float(elem.get("y1", 0) or 0)
            x2 = float(elem.get("x2", 0) or 0)
            y2 = float(elem.get("y2", 0) or 0)
            centerline = [(x1, y1), (x2, y2)]
        elif tag == "polyline":
            centerline = polygon_attr_to_vertices(elem.get("points", ""))
        
        if not centerline or len(centerline) < 2:
            continue
        
        # Transform to image coordinates
        combined_matrix = viewbox_to_image @ inherited_matrix
        centerline = transform_points(combined_matrix, centerline)
        
        # *** FIXED: Trim distance to avoid overlap with arrow head ***
        # Leave enough space for the arrow head (1.5x head size)
        trim_length_px = arrow_head_size_px
        
        # Calculate total path length
        total_length = 0.0
        for i in range(len(centerline) - 1):
            dx = centerline[i+1][0] - centerline[i][0]
            dy = centerline[i+1][1] - centerline[i][1]
            total_length += math.hypot(dx, dy)
        
        # Find trimmed endpoint for shaft
        shaft_centerline = centerline[:]
        shaft_end_point = centerline[-1]  # Will be updated below
        
        if total_length > trim_length_px:
            target_length = total_length - trim_length_px
            accumulated = 0.0
            shaft_centerline = [centerline[0]]
            
            for i in range(len(centerline) - 1):
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
                seg_length = math.hypot(dx, dy)
                
                if accumulated + seg_length < target_length:
                    shaft_centerline.append(centerline[i+1])
                    accumulated += seg_length
                else:
                    # Interpolate final point
                    remaining = target_length - accumulated
                    if seg_length > 0:
                        t = remaining / seg_length
                        x = centerline[i][0] + dx * t
                        y = centerline[i][1] + dy * t
                        shaft_end_point = (x, y)
                        shaft_centerline.append(shaft_end_point)
                    break
        else:
            # Path too short, use most of it for shaft
            mid_idx = max(1, len(centerline) - 2)
            shaft_centerline = centerline[:mid_idx+1]
            shaft_end_point = shaft_centerline[-1]
        
        # Create thin shaft polygon
        if len(shaft_centerline) >= 2:
            shaft_polygon = stroke_polyline_thin(shaft_centerline, arrow_shaft_thickness_px)
            if shaft_polygon and len(shaft_polygon) >= 3:
                annotations.append(Annotation(class_name="arrow", polygon_px=shaft_polygon))
        
        # *** FIXED: Create arrow head at TRUE endpoint (not trimmed point) ***
        # Arrow head should be at the actual end of the original path
        if len(centerline) >= 2:
            # Use the LAST point of original centerline
            end_point = centerline[-1]
            
            # Calculate direction from the point before the end
            # This gives us the correct arrow direction
            direction = (
                centerline[-1][0] - centerline[-2][0],
                centerline[-1][1] - centerline[-2][1]
            )
            
            head_triangle = arrow_head_triangle(end_point, direction, arrow_head_size_px)
            if head_triangle:
                annotations.append(Annotation(class_name="arrow_head", polygon_px=head_triangle))

    # Sort by class ID for consistent output
    annotations = sorted(annotations, key=lambda ann: CLASS_MAP[ann.class_name])
    return annotations, canvas_width, canvas_height


def export_yolo_labels(
    output_path: Path, 
    annotations: list[Annotation], 
    width: int, 
    height: int
) -> None:
    """Export annotations to YOLO segmentation format."""
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
    """Export annotations to Labelme JSON format for manual correction."""
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
    """Get BGR color for class ID."""
    palette = [
        (0, 255, 0),      # start - green
        (255, 0, 0),      # end - blue
        (0, 0, 255),      # inputoutput - red
        (0, 255, 255),    # operation - yellow
        (255, 255, 0),    # subroutine - cyan
        (255, 0, 255),    # condition - magenta
        (128, 255, 0),    # arrow - lime
        (255, 128, 0),    # arrow_head - orange
    ]
    return palette[class_id % len(palette)]


def draw_overlay(
    image_path: Path, 
    annotations: list[Annotation], 
    output_path: Path
) -> None:
    """Draw annotations overlay on image for visualization."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for ann in annotations:
        color = color_for_class(ann.class_id)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in ann.polygon_px], dtype=np.int32)
        
        if len(pts) >= 2:
            # Draw polygon outline
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            
            # Fill only arrow classes (to avoid obscuring node content)
            if ann.class_id in [CLASS_MAP["arrow"], CLASS_MAP["arrow_head"]]:
                if len(pts) >= 3:
                    cv2.fillPoly(image, [pts], color=(color[0]//6, color[1]//6, color[2]//6))

        # Draw label
        x1, y1, x2, y2 = [int(round(v)) for v in ann.bbox_px]
        cv2.putText(
            image,
            f"{ann.class_name}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    """Read image dimensions from file."""
    if not image_path or not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def process_file(
    svg_path: Path,
    flowcharts_dir: Path | None,
    images_dir: Path | None,
    output_dir: Path,
    labelme_dir: Path | None,
    visualize_dir: Path | None,
) -> bool:
    """Process single SVG file to YOLO annotations."""
    image_path = images_dir / f"{svg_path.stem}.png" if images_dir else None
    flowchart_path = flowcharts_dir / f"{svg_path.stem}.flowchart" if flowcharts_dir else None

    image_size = read_image_size(image_path) if image_path else None
    
    try:
        annotations, width, height = extract_annotations(
            svg_path, 
            flowchart_path=flowchart_path, 
            image_size=image_size
        )
    except Exception as e:
        print(f"Error processing {svg_path.name}: {e}")
        return False
    
    if not annotations:
        return False

    yolo_path = output_dir / f"{svg_path.stem}.txt"
    export_yolo_labels(yolo_path, annotations, width, height)

    if labelme_dir is not None:
        json_path = labelme_dir / f"{svg_path.stem}.json"
        export_labelme_json(
            json_path, 
            image_path if image_path else svg_path.with_suffix(".png"), 
            annotations, 
            width, 
            height
        )

    if visualize_dir is not None and image_path is not None:
        preview_path = visualize_dir / f"{svg_path.stem}.png"
        draw_overlay(image_path, annotations, preview_path)

    return True


def iter_svg_files(input_dir: Path) -> Iterable[Path]:
    """Iterate over all SVG files in directory."""
    return sorted(input_dir.rglob("*.svg"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert SVG flowchart annotations to YOLOv8 segmentation labels."
    )
    parser.add_argument("--input", type=Path, required=True, help="Folder containing SVG files")
    parser.add_argument("--output", type=Path, required=True, help="Folder for YOLO .txt labels")
    parser.add_argument("--flowcharts", type=Path, default=None, help="Optional folder with .flowchart files")
    parser.add_argument("--images", type=Path, default=None, help="Optional folder with PNG images")
    parser.add_argument("--labelme-output", type=Path, default=None, help="Optional folder for Labelme JSON output")
    parser.add_argument("--visualize", type=Path, default=None, help="Optional folder for overlay preview images")
    parser.add_argument("--max-files", type=int, default=0, help="Limit the number of SVG files processed (0 = all)")
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

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
    print(f"  Input folder     : {args.input}")
    print(f"  Output folder    : {args.output}")
    print(f"  Labelme folder   : {args.labelme_output}")
    print(f"  Visualize folder : {args.visualize}")
    print(f"  Processed        : {processed}")
    print(f"  Skipped          : {skipped}")


if __name__ == "__main__":
    main()