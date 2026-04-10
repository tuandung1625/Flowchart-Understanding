"""
For each sample (e.g. ID=42), reads:
  - {split}/flowchart/42.flowchart  -> node_id => node_type mapping
  - {split}/svg/42.svg              -> exact shape geometry (positions + sizes)

Outputs:
  - {split}/labels/42.txt           -> YOLO-seg format: class_id x_center y_center width height x1 y1 x2 y2 ... xn yn
                                       (all values normalized to [0,1] relative to image size)
                                       Polygon vertices preserve shape information (ovals, diamonds, parallelograms)

Class mapping:
  0 = start        (rounded rectangle, oval)
  1 = end          (rounded rectangle, oval)
  2 = inputoutput  (parallelogram)
  3 = operation    (plain rectangle)
  4 = subroutine   (rectangle with inner border lines)
  5 = condition    (diamond / rhombus)
"""

import os
import re
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
    """
    Extract vertices from SVG path by sampling points along the curve.
    For curves (bezier, arcs), sample evenly; for polygons, use actual vertices.
    Returns list of (x, y) tuples in local coordinates.
    """
    try:
        path = parse_path(d)
        vertices = []
        
        # Sample points along the path at even intervals
        for i in range(num_samples):
            t = i / max(num_samples - 1, 1)  # t in [0, 1]
            point = path.point(t)
            vertices.append((point.real, point.imag))
        
        return vertices if vertices else None
    except Exception:
        return None

# Extract rectangle corners as vertices
def rect_to_vertices(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

# Core SVG parser for YOLO-seg
def get_shape_annotations(
    svg_path: str,
    node_types: dict[str, str]
) -> dict[str, tuple[tuple[float, float, float, float], list[tuple[float, float]]]]:
    """
    Parse an SVG file and return normalised YOLO-seg annotations.
    
    Returns {node_id: ((x_center, y_center, width, height), [(x1, y1), (x2, y2), ...])} where:
    - bbox (x_center, y_center, width, height) in [0, 1]
    - polygon vertices in [0, 1] representing shape mask
    
    The SVG uses preserveAspectRatio="xMidYMid meet", so we apply the
    correct uniform-scale + centering offset before normalising.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # --- viewBox (the SVG coordinate space used by shape geometry) ---
    viewbox = root.get('viewBox', '').split()
    if len(viewbox) >= 4:
        vb_w, vb_h = float(viewbox[2]), float(viewbox[3])
    else:
        vb_w = float(root.get('width', 1))
        vb_h = float(root.get('height', 1))

    # --- viewport (the rendered canvas size, matches PNG at some integer scale) ---
    vp_w = float(root.get('width',  vb_w))
    vp_h = float(root.get('height', vb_h))

    # --- preserveAspectRatio="xMidYMid meet" transform ---
    # Scale the viewBox uniformly to fit inside the viewport, centred.
    scale    = min(vp_w / vb_w, vp_h / vb_h)
    offset_x = (vp_w - vb_w * scale) / 2
    offset_y = (vp_h - vb_h * scale) / 2

    annotations: dict[str, tuple[tuple[float, float, float, float], list[tuple[float, float]]]] = {}

    for elem in root.iter():
        elem_id = elem.get('id', '')

        # Only process elements whose id directly maps to a flowchart node.
        # This automatically skips text labels (ids ending in 't') and the
        # inner border rect of subroutine shapes (ids ending in 'i').
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
            bbox_info = (x, y, w, h)
            vertices = rect_to_vertices(x, y, w, h)

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


# -------------------------------------------------------------------
# Annotation writer
# -------------------------------------------------------------------

def generate_yolo_annotation(
    flowchart_path: str,
    svg_path: str,
    output_path: str
) -> bool:
    """
    Generate a YOLO-seg .txt annotation file for one sample.
    Format: class_id x_center y_center width height x1 y1 x2 y2 ... xn yn
    Returns True on success, False if no annotations were produced.
    """
    node_types = parse_flowchart(flowchart_path)
    annotations = get_shape_annotations(svg_path, node_types)

    lines = []
    for node_id, (bbox, vertices) in annotations.items():
        node_type = node_types.get(node_id)
        if node_type not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[node_type]
        xc, yc, w, h = bbox
        
        # Build polygon vertex string
        vertex_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in vertices)
        
        # YOLO-seg format: class_id bbox polygon
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {vertex_str}")

    if not lines:
        return False

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return True


# -------------------------------------------------------------------
# Dataset processing
# -------------------------------------------------------------------

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
# Auto-generated by generate_yolo_annotations.py
# Format: class_id x_center y_center width height x1 y1 x2 y2 ... xn yn

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


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == '__main__':
    print("=== FloCo → YOLO Annotation Generator ===")
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