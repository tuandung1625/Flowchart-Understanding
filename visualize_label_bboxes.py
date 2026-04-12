import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


@dataclass
class Annotation:
    class_id: int
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: list[tuple[int, int]]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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


def norm_xy_to_px(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    px = int(round(clamp01(x) * (width - 1)))
    py = int(round(clamp01(y) * (height - 1)))
    return px, py


def yolo_bbox_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = clamp01(xc - w / 2)
    y1 = clamp01(yc - h / 2)
    x2 = clamp01(xc + w / 2)
    y2 = clamp01(yc + h / 2)

    p1 = norm_xy_to_px(x1, y1, width, height)
    p2 = norm_xy_to_px(x2, y2, width, height)
    return p1[0], p1[1], p2[0], p2[1]


def polygon_to_xyxy(points: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    px_points = [norm_xy_to_px(x, y, width, height) for x, y in points]
    xs = [p[0] for p in px_points]
    ys = [p[1] for p in px_points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_label_line(parts: list[float], width: int, height: int, label_format: str) -> Annotation | None:
    if len(parts) < 5:
        return None

    class_id = int(parts[0])
    nums = parts[1:]

    polygon: list[tuple[float, float]] = []

    if label_format == "bbox":
        if len(nums) != 4:
            return None
        xc, yc, w, h = nums
        bbox = yolo_bbox_to_xyxy(xc, yc, w, h, width, height)
        return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=[])

    if label_format == "seg":
        if len(nums) < 6 or len(nums) % 2 != 0:
            return None
        polygon = list(zip(nums[0::2], nums[1::2]))
        bbox = polygon_to_xyxy(polygon, width, height)
        polygon_px = [norm_xy_to_px(x, y, width, height) for x, y in polygon]
        return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=polygon_px)

    if label_format == "seg+bbox":
        if len(nums) < 10 or (len(nums) - 4) % 2 != 0:
            return None
        xc, yc, w, h = nums[:4]
        polygon = list(zip(nums[4::2], nums[5::2]))
        bbox = yolo_bbox_to_xyxy(xc, yc, w, h, width, height)
        polygon_px = [norm_xy_to_px(x, y, width, height) for x, y in polygon]
        return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=polygon_px)

    # auto mode
    if len(nums) == 4:
        xc, yc, w, h = nums
        bbox = yolo_bbox_to_xyxy(xc, yc, w, h, width, height)
        return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=[])

    if len(nums) >= 10 and (len(nums) - 4) % 2 == 0:
        xc, yc, w, h = nums[:4]
        if is_probable_yolo_bbox(xc, yc, w, h):
            polygon = list(zip(nums[4::2], nums[5::2]))
            bbox = yolo_bbox_to_xyxy(xc, yc, w, h, width, height)
            polygon_px = [norm_xy_to_px(x, y, width, height) for x, y in polygon]
            return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=polygon_px)

    if len(nums) >= 6 and len(nums) % 2 == 0:
        polygon = list(zip(nums[0::2], nums[1::2]))
        bbox = polygon_to_xyxy(polygon, width, height)
        polygon_px = [norm_xy_to_px(x, y, width, height) for x, y in polygon]
        return Annotation(class_id=class_id, bbox_xyxy=bbox, polygon_xy=polygon_px)

    return None


def color_for_class(class_id: int) -> tuple[int, int, int]:
    # Stable distinct-ish colors in BGR
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


def dim_color(color: tuple[int, int, int], factor: float = 0.5) -> tuple[int, int, int]:
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    return sorted(labels_dir.rglob("*.txt"))


def visualize_labels(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    label_format: str,
    max_images: int,
    draw_bbox: bool,
    draw_polygon: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(iter_label_files(labels_dir))
    if max_images > 0:
        label_files = label_files[:max_images]

    total = 0
    missing_images = 0
    parse_errors = 0

    for label_path in label_files:
        stem = label_path.stem
        image_path = images_dir / f"{stem}.png"
        if not image_path.exists():
            missing_images += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            parse_errors += 1
            continue

        h, w = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            try:
                parts = [float(x) for x in line.split()]
                ann = parse_label_line(parts, w, h, label_format)
                if ann is None:
                    parse_errors += 1
                    continue

                x1, y1, x2, y2 = ann.bbox_xyxy
                color = color_for_class(ann.class_id)
                bbox_color = dim_color(color, 0.45)
                if draw_bbox:
                    cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 1)
                cv2.putText(
                    image,
                    f"cls:{ann.class_id}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    bbox_color if draw_bbox else color,
                    1,
                    cv2.LINE_AA,
                )

                if draw_polygon and ann.polygon_xy:
                    pts = ann.polygon_xy
                    for i in range(len(pts)):
                        p1 = pts[i]
                        p2 = pts[(i + 1) % len(pts)]
                        cv2.line(image, p1, p2, color, 2)

                    # Mark polygon vertices to make non-rectangular shapes obvious.
                    for p in pts:
                        cv2.circle(image, p, 2, color, -1)

            except ValueError:
                parse_errors += 1

        out_path = output_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), image)
        total += 1

    print("Visualization complete")
    print(f"  Labels folder : {labels_dir}")
    print(f"  Images folder : {images_dir}")
    print(f"  Output folder : {output_dir}")
    print(f"  Images drawn  : {total}")
    print(f"  Missing images: {missing_images}")
    print(f"  Parse issues  : {parse_errors}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw YOLO bounding boxes from label files onto corresponding PNG images.",
    )
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels folder")
    parser.add_argument("--images", type=Path, required=True, help="Path to PNG images folder")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for visualized images")
    parser.add_argument(
        "--label-format",
        choices=["auto", "bbox", "seg", "seg+bbox"],
        default="auto",
        help="Expected label format",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument(
        "--polygon-only",
        action="store_true",
        help="Draw only polygon outlines/points (hide bounding boxes)",
    )
    parser.add_argument(
        "--no-polygon",
        action="store_true",
        help="Do not draw polygon outlines",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    visualize_labels(
        labels_dir=args.labels,
        images_dir=args.images,
        output_dir=args.output,
        label_format=args.label_format,
        max_images=args.max_images,
        draw_bbox=not args.polygon_only,
        draw_polygon=not args.no_polygon,
    )


if __name__ == "__main__":
    main()
