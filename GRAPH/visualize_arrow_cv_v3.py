"""Visualize CV-detected connectors on top of the source image (v3)."""
# python GRAPH/visualize_arrow_cv_v3.py --image DATASET/Train/images/8.png --connectors runs/arrow_v3/8.connectors.json --output runs/arrow_v3_vis --draw-bbox --draw-id 
 
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize connectors from arrow_cv_v3 output JSON")
    p.add_argument("--image", required=True, help="Path to source image")
    p.add_argument("--connectors", required=True, help="Path to *.connectors.v3.json")
    p.add_argument("--output", default="runs/arrow_v3_vis", help="Output image path or folder")
    p.add_argument("--draw-bbox", action="store_true", help="Draw connector bounding boxes")
    p.add_argument("--draw-id", action="store_true", help="Draw connector IDs")
    p.add_argument("--thickness", type=int, default=2, help="Polyline thickness")
    return p.parse_args()


def resolve_output_path(output_arg: str, image_path: Path) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{image_path.stem}.arrow_cv_v3.vis.png"


def load_connectors(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("connectors", [])


def color_for_idx(idx: int) -> tuple[int, int, int]:
    # Deterministic high-contrast palette in BGR for OpenCV
    palette = [
        (0, 255, 255),
        (0, 200, 0),
        (255, 180, 0),
        (255, 0, 180),
        (0, 140, 255),
        (220, 220, 0),
        (0, 255, 120),
        (255, 120, 120),
    ]
    return palette[idx % len(palette)]


def draw_polyline(img: np.ndarray, polyline: list[list[float]], color: tuple[int, int, int], thickness: int) -> None:
    if len(polyline) < 2:
        return
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polyline], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    connectors_path = Path(args.connectors)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not connectors_path.exists():
        raise FileNotFoundError(f"Connectors JSON not found: {connectors_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    connectors = load_connectors(connectors_path)

    # Overlay layer so lines are vivid but underlying image is still visible
    overlay = img.copy()

    for i, c in enumerate(connectors):
        color = color_for_idx(i)
        poly = c.get("polyline", [])
        draw_polyline(overlay, poly, color=color, thickness=max(1, args.thickness + 1))

        center = c.get("center", None)
        if center and len(center) == 2:
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(overlay, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)

        if args.draw_bbox:
            bbox = c.get("bbox_xyxy", None)
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        if args.draw_id:
            cid = str(c.get("connector_id", f"c_{i+1}"))
            anchor = None
            if center and len(center) == 2:
                anchor = (int(round(center[0])) + 5, int(round(center[1])) - 5)
            elif poly:
                p0 = poly[0]
                anchor = (int(round(p0[0])) + 5, int(round(p0[1])) - 5)
            if anchor:
                cv2.putText(overlay, cid, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.7, img, 0.3, 0.0)

    out_path = resolve_output_path(args.output, image_path)
    ok = cv2.imwrite(str(out_path), vis)
    if not ok:
        raise RuntimeError(f"Failed to save visualization: {out_path}")

    print(f"Visualization saved: {out_path}")
    print(f"Total connectors drawn: {len(connectors)}")


if __name__ == "__main__":
    main()
