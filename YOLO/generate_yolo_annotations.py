import os
import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from svgpathtools import parse_path

CLASS_MAP = {
    'start':       0,
    'end':         1,
    'inputoutput': 2,
    'operation':   3,
    'subroutine':  4,
    'condition':   5,
}

DATASET_ROOT = Path(r"d:\Workspaces\PROJECT\Thesis - Flowchart\DATASET")
SPLITS = ['Train', 'Validation', 'Test']

# Parse a flowchart file and return {node_id: node_type}
def parse_flowchart(path: str) -> dict[str, str]:
    node_types = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            # Declaration format:  id=>type: label text
            m = re.match(r'^(\w+)=>(\w+):', line.strip())
            if m:
                node_types[m.group(1)] = m.group(2)
    return node_types

# Extract (tx, ty) translation from 'matrix(a,b,c,d,tx,ty)'
def parse_matrix_transform(transform_str: str) -> tuple[float, float]:
    if not transform_str:
        return 0.0, 0.0
    m = re.search(
        r'matrix\(\s*'
        r'[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,'
        r'\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)',
        transform_str
    )
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

# Returns (x, y, width, height) in local (pre-transform) coordinates.
# Handles all SVG path commands: M (move), L (line), C (cubic bezier),
# Q (quadratic), A (arc), Z (close), etc.
def path_bbox(d: str) -> tuple[float, float, float, float] | None:
    try:
        path = parse_path(d)
        bbox = path.bbox()
        if bbox is None or len(bbox) != 4:
            return None
        x_min, x_max, y_min, y_max = bbox
        return x_min, y_min, x_max - x_min, y_max - y_min
    except Exception:
        # Fallback: if parsing fails, return None
        return None

# Extract polygon vertices from SVG path (sampled points along the curve)
def path_to_vertices(d: str, num_samples: int = 20) -> list[tuple[float, float]] | None:
    try:
        path = parse_path(d)
        if not path:
            return None

        # If the path is polygonal (line-only), preserve exact corners.
        if all(seg.__class__.__name__ == 'Line' for seg in path):
            vertices = [(path[0].start.real, path[0].start.imag)]
            for seg in path:
                end_pt = (seg.end.real, seg.end.imag)
                if end_pt != vertices[-1]:
                    vertices.append(end_pt)
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices.pop()
            return vertices

        # For curves/arcs, sample each segment so non-linear shapes are preserved.
        vertices = []
        seg_samples = max(6, num_samples // max(1, len(path)))
        for seg in path:
            if not vertices:
                vertices.append((seg.start.real, seg.start.imag))
            for i in range(1, seg_samples + 1):
                t = i / seg_samples
                point = seg.point(t)
                pt = (point.real, point.imag)
                if pt != vertices[-1]:
                    vertices.append(pt)

        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()

        return vertices if vertices else None
    except Exception:
        return None

# Extract rectangle corners as vertices
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

    # SVG behavior: if one of rx/ry is zero but the other is non-zero, they mirror.
    if rx == 0.0 and ry > 0.0:
        rx = min(ry, w / 2)
    if ry == 0.0 and rx > 0.0:
        ry = min(rx, h / 2)

    if rx == 0.0 and ry == 0.0:
        return [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ]

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
        for i in range(1, steps + 1):
            t = i / steps
            ang = start_angle + (end_angle - start_angle) * t
            pts.append((cx + arx * math.cos(ang), cy + ary * math.sin(ang)))
        return pts

    n = max(3, arc_samples)
    vertices = [
        (x + rx, y),
        (x + w - rx, y),
    ]
    vertices.extend(arc_points(x + w - rx, y + ry, rx, ry, -math.pi / 2, 0.0, n))
    vertices.append((x + w, y + h - ry))
    vertices.extend(arc_points(x + w - rx, y + h - ry, rx, ry, 0.0, math.pi / 2, n))
    vertices.append((x + rx, y + h))
    vertices.extend(arc_points(x + rx, y + h - ry, rx, ry, math.pi / 2, math.pi, n))
    vertices.append((x, y + ry))
    vertices.extend(arc_points(x + rx, y + ry, rx, ry, math.pi, 3 * math.pi / 2, n))

    cleaned: list[tuple[float, float]] = []
    for pt in vertices:
        if not cleaned or pt != cleaned[-1]:
            cleaned.append(pt)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned

# Core SVG parser for YOLO-seg
def get_shape_annotations(
    svg_path: str,
    node_types: dict[str, str]
) -> dict[str, tuple[tuple[float, float, float, float], list[tuple[float, float]]]]:
    """
    Parse an SVG file and return normalised YOLO-seg annotations.
    
    Returns {node_id: ((x_center, y_center, width, height), [(x1, y1), (x2, y2), ...])} where:
    
    The SVG uses preserveAspectRatio="xMidYMid meet", so we apply the
    correct uniform-scale + centering offset before normalising.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # viewBox (the SVG coordinate space used by shape geometry)
    viewbox = root.get('viewBox', '').split()
    if len(viewbox) >= 4:
        vb_w, vb_h = float(viewbox[2]), float(viewbox[3])
    else:
        vb_w = float(root.get('width', 1))
        vb_h = float(root.get('height', 1))

    # viewport (the rendered canvas size, matches PNG at some integer scale)
    vp_w = float(root.get('width',  vb_w))
    vp_h = float(root.get('height', vb_h))

    # preserveAspectRatio="xMidYMid meet" transform
    # Scale the viewBox uniformly to fit inside the viewport, centred.
    scale    = min(vp_w / vb_w, vp_h / vb_h)
    offset_x = (vp_w - vb_w * scale) / 2
    offset_y = (vp_h - vb_h * scale) / 2

    annotations: dict[str, tuple[tuple[float, float, float, float], list[tuple[float, float]]]] = {}

    for elem in root.iter():
        elem_id = elem.get('id', '')

        # Only process elements whose id directly maps to a flowchart node.
        if elem_id not in node_types:
            continue

        tag = elem.tag.split('}')[-1]   # strip XML namespace prefix
        tx, ty = parse_matrix_transform(elem.get('transform', ''))
        
        vertices = []
        bbox_info = None

        if tag == 'rect':
            x = float(elem.get('x', 0)) + tx
            y = float(elem.get('y', 0)) + ty
            w = float(elem.get('width',  0))
            h = float(elem.get('height', 0))
            rx = float(elem.get('rx', 0) or 0)
            ry = float(elem.get('ry', 0) or 0)
            bbox_info = (x, y, w, h)
            vertices = rect_to_vertices(x, y, w, h, rx=rx, ry=ry)

        elif tag == 'path':
            # Skip connector arrows (fill="none")
            if elem.get('fill', '') == 'none':
                continue
            
            # Get bbox for bounding box
            bbox_result = path_bbox(elem.get('d', ''))
            if bbox_result is None:
                continue
            px, py, pw, ph = bbox_result
            bbox_info = (px + tx, py + ty, pw, ph)
            
            # Get sampled vertices for polygon mask
            path_vertices = path_to_vertices(elem.get('d', ''), num_samples=20)
            if path_vertices:
                # Apply transform
                vertices = [(vx + tx, vy + ty) for vx, vy in path_vertices]

        else:
            continue   # ignore any other element types

        if not bbox_info or not vertices:
            continue

        x, y, w, h = bbox_info
        
        # --- Convert viewBox coords -> normalised [0,1] (image space) ---
        x_center_norm = ((x + w / 2) * scale + offset_x) / vp_w
        y_center_norm = ((y + h / 2) * scale + offset_y) / vp_h
        w_norm        = (w * scale) / vp_w
        h_norm        = (h * scale) / vp_h

        # Clamp bbox to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        w_norm        = max(1e-4, min(1.0, w_norm))
        h_norm        = max(1e-4, min(1.0, h_norm))
        
        # Normalize polygon vertices
        normalized_vertices = []
        for vx, vy in vertices:
            vx_norm = (vx * scale + offset_x) / vp_w
            vy_norm = (vy * scale + offset_y) / vp_h
            vx_norm = max(0.0, min(1.0, vx_norm))
            vy_norm = max(0.0, min(1.0, vy_norm))
            normalized_vertices.append((vx_norm, vy_norm))

        annotations[elem_id] = ((x_center_norm, y_center_norm, w_norm, h_norm), normalized_vertices)

    return annotations

# Annotation writer
def generate_yolo_annotation(
    flowchart_path: str,
    svg_path: str,
    output_path: str
) -> bool:
    """
    Generate a YOLO-seg .txt annotation file for one sample.
    Ultralytics YOLOv8-seg format: class_id x1 y1 x2 y2 ... xn yn
    """
    node_types = parse_flowchart(flowchart_path)
    annotations = get_shape_annotations(svg_path, node_types)

    lines = []
    for node_id, (_, vertices) in annotations.items():
        node_type = node_types.get(node_id)
        if node_type not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[node_type]

        # Build polygon vertex string
        vertex_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in vertices)

        # YOLOv8-seg format: class_id polygon
        lines.append(f"{class_id} {vertex_str}")

    if not lines:
        return False

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return True

# Dataset processing
def process_split(split_dir: Path) -> None:
    flowchart_dir = split_dir / 'flowchart'
    svg_dir       = split_dir / 'svg'
    labels_dir    = split_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)

    files = sorted(flowchart_dir.glob('*.flowchart'))
    if not files:
        print(f"  No .flowchart files found in {flowchart_dir}")
        return

    print(f"\n[{split_dir.name}] Processing {len(files)} samples ...")

    ok = errors = skipped = 0
    for fc_path in files:
        sid      = fc_path.stem
        svg_path = svg_dir / f"{sid}.svg"
        out_path = labels_dir / f"{sid}.txt"

        if not svg_path.exists():
            skipped += 1
            continue
        try:
            if generate_yolo_annotation(str(fc_path), str(svg_path), str(out_path)):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR on sample {sid}: {e}")
            errors += 1

    print(f"  Generated: {ok}  |  Skipped (no SVG / no shapes): {skipped}  |  Errors: {errors}")
    print(f"  Labels written to: {labels_dir}")


def write_yaml(dataset_root: Path) -> None:
    """Write a YOLO-seg dataset.yaml file for use with Ultralytics YOLOv8."""
    class_names = sorted(CLASS_MAP, key=CLASS_MAP.get)
    yaml_content = f"""\
# FloCo flowchart shape detection dataset (YOLO-seg segmentation format)
path: {dataset_root.as_posix()}

train: Train/png
val:   Validation/png
test:  Test/png

nc: {len(CLASS_MAP)}
names: {class_names}
"""
    yaml_path = dataset_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset YAML (YOLO-seg) written to: {yaml_path}")

# RUN
if __name__ == '__main__':
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Class map    : {CLASS_MAP}\n")

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            process_split(split_dir)
        else:
            print(f"[{split}] Directory not found, skipping.")

    write_yaml(DATASET_ROOT)
    print("\nDone.")