"""Final retrain on champion self-labels (321 wheels / 239 frames).
yolo11s-pose + heavy aug + fine-tune from soft_s_aug.
"""

from ultralytics import YOLO

SEED = "runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt"

model = YOLO(SEED)
model.train(
    data="configs/pose_dataset_real_v1_self.yaml",
    epochs=100,
    device="mps",
    project="runs/pose",
    name="wheel_real_v1_self_s",
    mosaic=1.0,
    mixup=0.20,
    copy_paste=0.4,
    hsv_h=0.020,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.10,
    scale=0.5,
    fliplr=0.5,
    degrees=15.0,
    patience=40,
    cos_lr=True,
    lr0=0.005,
)
