# RUN 
# python YOLO\eval_yolo_v2.py

from ultralytics import YOLO
import os

model = YOLO("runs/segment/runs/flowchart_seg_v2_exp1/weights/best.pt")

# Run validation with all plot options
results = model.val(
    data="DATASET/dataset.yaml",
    task="segment",
    imgsz=512,
    batch=1,
    device="cpu",  # Use CPU since no CUDA available
    plots=True,
    save_json=True,
    conf=0.25,  # Confidence threshold
    iou=0.6     # IoU threshold
)

print("\n" + "="*50)
print("Validation finished!")
print("="*50)

# Show where plots are saved
results_dir = results.save_dir
print(f"\nResults saved to: {results_dir}")
print("\nGenerated files:")
for file in os.listdir(results_dir):
    print(f"  - {file}")
    
# List the key plot files
print("\nKey plots to check:")
print(f"  - confusion_matrix.png (Class confusion matrix)")
print(f"  - confusion_matrix_normalized.png")
print(f"  - F1_curve.png, P_curve.png, R_curve.png")
print(f"  - PR_curve.png (Precision-Recall curve)")
if "masks" in str(results_dir).lower() or "segment" in str(results_dir).lower():
    print(f"  - masks/ (Segmentation masks)")
    print(f"  - Box PR curves for each class")

# yolo val model=runs/segment/runs/flowchart_seg_v2_exp1/weights/best.pt data=DATASET/dataset.yaml task=segment imgsz=512 batch=1 plots=True save_json=True