"""Training loop skeleton for the MobileNetV2-skipless wheel pose detector.

Smoke target: runs 2 epochs on synthetic random data, checkpoints to
runs/pose_mn2/<name>/weights/. This is *not* a production-ready
training script — the data loader emits noise, not labelled real
frames. Wiring is correct end-to-end (model + matcher + loss + opt +
sched + checkpoint), so when real data arrives the only piece to
swap is `SyntheticBatch` for a real `Dataset`.

Why a skeleton not a full trainer:
- We have no real labelled batch yet (CLAUDE.md "Current blockers").
- AR-team has not signed off on latency / app-size / INT8 budgets
  (docs/MODEL_ARCHITECTURE_PROPOSAL.md §7).
- Premature optimization to write a 60-epoch trainer when both data
  shape and budget targets are still open.

When data arrives: replace `_synthetic_batch` with a Dataset that
yields (image_tensor, gt_bboxes, gt_keypoints, gt_visibility) per
sample, following the plugin-format coordinates in pixel space.

Usage:
    python scripts/train_mobilenetv2_skipless.py --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.loss import PoseLoss  # noqa: E402
from src.models.matcher import assign_targets_batched  # noqa: E402
from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    FEATURE_STRIDE,
    MobileNetV2SkiplessPose,
    N_KEYPOINTS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV2-skipless wheel pose detector (skeleton)"
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument(
        "--device",
        default="cpu",
        help="'cpu', 'mps', or '0' for CUDA. Synthetic-data smoke can run on any.",
    )
    p.add_argument(
        "--project",
        type=Path,
        default=REPO / "runs" / "pose_mn2",
        help="Run output dir (analogous to runs/pose/ for the Ultralytics path).",
    )
    p.add_argument("--name", default="smoke")
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialise encoder from torchvision ImageNet weights (needs internet).",
    )
    return p.parse_args()


def _synthetic_batch(
    batch_size: int, imgsz: int = 640
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Random images + random small wheels for smoke-testing the pipeline.

    Each image gets 1-3 random wheels with bboxes well inside the
    image bounds. Keypoints are placed at predictable locations
    relative to the bbox (left rim, right rim, lowest disc point) so
    visibility is always 1.
    """
    images = torch.rand(batch_size, 3, imgsz, imgsz)
    gt_bboxes: list[torch.Tensor] = []
    gt_keypoints: list[torch.Tensor] = []
    gt_visibility: list[torch.Tensor] = []
    for _ in range(batch_size):
        n = int(torch.randint(1, 4, (1,)).item())
        bboxes = torch.empty(n, 4)
        keypoints = torch.empty(n, N_KEYPOINTS, 2)
        for i in range(n):
            w = float(torch.randint(40, 200, (1,)).item())
            h = float(torch.randint(40, 200, (1,)).item())
            x1 = float(torch.randint(0, imgsz - int(w), (1,)).item())
            y1 = float(torch.randint(0, imgsz - int(h), (1,)).item())
            x2 = x1 + w
            y2 = y1 + h
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bboxes[i] = torch.tensor([x1, y1, x2, y2])
            keypoints[i, 0] = torch.tensor([x1 + 0.1 * w, cy])  # A
            keypoints[i, 1] = torch.tensor([x2 - 0.1 * w, cy])  # B
            keypoints[i, 2] = torch.tensor([cx, y2 - 0.05 * h])  # C
        gt_bboxes.append(bboxes)
        gt_keypoints.append(keypoints)
        gt_visibility.append(torch.ones(n, N_KEYPOINTS))
    return images, gt_bboxes, gt_keypoints, gt_visibility


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    out_dir = args.project / args.name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = MobileNetV2SkiplessPose(pretrained=args.pretrained).to(device)
    criterion = PoseLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch, eta_min=1e-5
    )

    imgsz = 640
    grid_hw = (imgsz // FEATURE_STRIDE, imgsz // FEATURE_STRIDE)

    log_lines: list[str] = []
    print(f"device:         {device}")
    print(f"epochs × steps: {args.epochs} × {args.steps_per_epoch}")
    print(f"batch size:     {args.batch}")
    print(f"out dir:        {out_dir}")
    print(f"pretrained enc: {args.pretrained}")
    print()

    started_at = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            images, gt_bb, gt_kp, gt_v = _synthetic_batch(args.batch, imgsz)
            images = images.to(device)
            gt_bb = [b.to(device) for b in gt_bb]
            gt_kp = [k.to(device) for k in gt_kp]
            gt_v = [v.to(device) for v in gt_v]

            preds = model(images)
            match = assign_targets_batched(gt_bb, gt_kp, gt_v, grid_hw, FEATURE_STRIDE)
            ld = criterion(preds, match)

            optimizer.zero_grad()
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(ld["total"].detach())

            line = (
                f"epoch {epoch}/{args.epochs} step {step + 1}/{args.steps_per_epoch} "
                f"total={float(ld['total'].detach()):.4f} "
                f"cls={float(ld['cls']):.4f} "
                f"bbox={float(ld['bbox']):.4f} "
                f"kpt={float(ld['kpt']):.4f} "
                f"vis={float(ld['vis']):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
            print(line)
            log_lines.append(line)

        avg = epoch_loss / max(args.steps_per_epoch, 1)
        print(f"  → epoch {epoch} avg total loss: {avg:.4f}")

    duration = time.time() - started_at
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "config": vars(args) | {"imgsz": imgsz, "grid_hw": grid_hw},
    }
    ckpt_path = weights_dir / "last.pt"
    torch.save(ckpt, ckpt_path)
    (out_dir / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print(f"\nduration: {duration:.1f}s")
    print(f"checkpoint: {ckpt_path}")
    print(f"NOTE: this trained on SYNTHETIC random data. Not a usable model.")


if __name__ == "__main__":
    main()
