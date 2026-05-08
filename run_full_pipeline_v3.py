#!/usr/bin/env python3
"""
End-to-end v3 pipeline (YOLO-6 nodes only + CV arrow detection).
Runs: OCR (v3) → CV Arrow Detection → Build Graph (v3) → Attach Floating Text

Usage:
    python run_full_pipeline_v3.py DATASET/Test/images/10103.png --model YOLO/runs/results/weights/best.pt --output runs/full_pipeline
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, step_name):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n❌ FAILED at {step_name}")
        sys.exit(1)
    print(f"✅ {step_name} completed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run v3 OCR pipeline (YOLO-6 + CV arrows) end-to-end for one image"
    )
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--model", type=str, required=True, help="YOLO model (.pt, 6-class)")
    parser.add_argument("--names", type=str, default="", help="Dataset YAML or names")
    parser.add_argument("--output", type=str, default="runs/pipeline_v3_output", help="Base output folder")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for text classification")
    parser.add_argument("--ocr-scale", type=float, default=1.0, help="Scale factor for OCR")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Min confidence for postprocessing")
    parser.add_argument("--distance-threshold", type=float, default=50.0, help="Distance threshold for attaching text to edges")
    parser.add_argument("--arrow-min-length", type=float, default=40.0, help="Min arrow segment length for CV detector")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip postprocessing step")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    output_base = Path(args.output)
    ocr_output = output_base / "ocr_v3"
    post_output = output_base / "postprocess_v3"
    arrows_output = output_base / "arrows_v3"
    graph_output = output_base / "graph_v3"
    final_output = output_base / "final_v3"

    # Step 1: Run OCR on full image (v3 - YOLO-6 only)
    step1_cmd = [
        sys.executable, "OCR_v3/ocr_full_image_v3.py",
        str(image_path),
        "--model", str(model_path),
        "--output", str(ocr_output),
        "--iou-threshold", str(args.iou_threshold),
        "--ocr-scale", str(args.ocr_scale),
    ]
    if args.names:
        step1_cmd.extend(["--names", args.names])
    run_command(step1_cmd, "Full-Image OCR (v3 - YOLO-6)")

    # Step 2: Postprocess (optional)
    if not args.skip_postprocess:
        step2_cmd = [
            sys.executable, "OCR_v3/postprocess_ocr_full.py",
            "--input", str(ocr_output),
            "--output", str(post_output),
            "--min-conf", str(args.min_conf),
            "--normalize",
            "--merge-node-texts",
        ]
        run_command(step2_cmd, "Postprocessing")
        graph_input = post_output
    else:
        print("\n⏭️  Skipping postprocessing, using raw OCR output")
        graph_input = ocr_output

    # Step 3: Detect arrows via CV (v3)
    step3_cmd = [
        sys.executable, "GRAPH/arrow_cv_v3.py",
        str(image_path),
        "--output", str(arrows_output),
        "--min-length", str(args.arrow_min_length),
    ]
    run_command(step3_cmd, "Arrow Detection (CV-based, v3)")

    # Step 4: Build graph from OCR output.
    # build_graph_v2.py currently infers connectors internally from the OCR JSON schema,
    # so the CV arrow output remains a separate intermediate artifact for now.
    step4_cmd = [
        sys.executable, "GRAPH/build_graph_v2.py",
        "--input", str(graph_input),
        "--output", str(graph_output),
    ]
    run_command(step4_cmd, "Build Graph (with CV arrows, v3)")

    # Step 5: Attach floating text to edges
    step5_cmd = [
        sys.executable, "OCR/attach_floating_text_to_edges.py",
        "--ocr-input", str(graph_input),
        "--graph-input", str(graph_output),
        "--output", str(final_output),
        "--distance-threshold", str(args.distance_threshold),
    ]
    run_command(step5_cmd, "Attach Floating Text to Edges")

    # Summary
    print(f"\n{'='*60}")
    print("✅ V3 PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"📁 Output files:")
    print(f"   OCR output:      {ocr_output}")
    print(f"   Postprocess:     {post_output}")
    print(f"   Arrows detected: {arrows_output}")
    print(f"   Graph output:    {graph_output}")
    print(f"   Final output:    {final_output}")

    # List output files
    if final_output.exists():
        outputs = list(final_output.glob("*.json"))
        if outputs:
            print(f"\n📄 Generated files:")
            for f in outputs:
                print(f"   - {f.name}")
                # Try to show a sample of the output
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    if "node_count" in data:
                        print(f"     Nodes: {data.get('node_count')}")
                    if "edge_count" in data:
                        print(f"     Edges: {data.get('edge_count')}")
                    if "floating_text_stats" in data:
                        stats = data["floating_text_stats"]
                        print(f"     Floating texts: {stats.get('total')} (attached: {stats.get('attached')})")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
