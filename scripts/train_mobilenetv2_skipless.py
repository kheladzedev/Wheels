"""Train the MobileNetV2-skipless wheel pose detector.

Without ``--dataset-root`` this keeps the original synthetic smoke path.
With ``--dataset-root`` it reads the YOLO-pose datasets emitted by the
Unreal acceptance pipeline:

    images/{train,val}/*.jpg
    labels/{train,val}/*.txt

This real-data path is meant for provisional baselines until the Unreal
batch passes the data-quality gate; it does not make a production-model
claim by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.loss import PoseLoss  # noqa: E402
from src.models.matcher import MatchResult, assign_targets_batched  # noqa: E402
from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    FEATURE_STRIDE,
    MobileNetV2SkiplessPose,
    N_KEYPOINTS,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetStats:
    image_count: int
    labelled_wheel_count: int
    empty_label_count: int


class YoloPoseDataset(Dataset):
    """YOLO-pose dataset reader returning pixel-space tensors.

    Label format per line:
        class cx cy w h ax ay av bx by bv cx cy cv

    Coordinates are normalized to the resized square image. Visibility is
    converted to a 0/1 target so empty or negative images remain valid.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        imgsz: int,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"missing image split directory: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"missing label split directory: {self.labels_dir}")

        images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if limit is not None:
            images = images[:limit]
        self.image_paths = images
        self.stats = self._scan_stats()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
        )
        image_np = image_rgb.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        label_path = self._label_path_for(image_path)
        bboxes, keypoints, visibility = self._read_label(label_path)
        return image_tensor, bboxes, keypoints, visibility

    def _label_path_for(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _scan_stats(self) -> DatasetStats:
        labelled = 0
        empty = 0
        for image_path in self.image_paths:
            label_path = self._label_path_for(image_path)
            if not label_path.exists() or label_path.stat().st_size == 0:
                empty += 1
                continue
            lines = [
                ln
                for ln in label_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if lines:
                labelled += len(lines)
            else:
                empty += 1
        return DatasetStats(
            image_count=len(self.image_paths),
            labelled_wheel_count=labelled,
            empty_label_count=empty,
        )

    def _read_label(
        self, label_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not label_path.exists() or label_path.stat().st_size == 0:
            return _empty_targets()

        bbox_rows: list[list[float]] = []
        keypoint_rows: list[list[list[float]]] = []
        visibility_rows: list[list[float]] = []
        for line_no, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split()]
            expected = 5 + N_KEYPOINTS * 3
            if len(values) != expected:
                raise ValueError(
                    f"{label_path}:{line_no} expected {expected} YOLO-pose fields, "
                    f"got {len(values)}"
                )

            _, cx, cy, w, h, *kpt_values = values
            px_cx = cx * self.imgsz
            px_cy = cy * self.imgsz
            px_w = w * self.imgsz
            px_h = h * self.imgsz
            x1 = max(0.0, px_cx - px_w / 2.0)
            y1 = max(0.0, px_cy - px_h / 2.0)
            x2 = min(float(self.imgsz), px_cx + px_w / 2.0)
            y2 = min(float(self.imgsz), px_cy + px_h / 2.0)
            bbox_rows.append([x1, y1, x2, y2])

            kpts: list[list[float]] = []
            vis: list[float] = []
            for k in range(N_KEYPOINTS):
                base = k * 3
                kx = kpt_values[base] * self.imgsz
                ky = kpt_values[base + 1] * self.imgsz
                kv = kpt_values[base + 2]
                kpts.append([kx, ky])
                vis.append(1.0 if kv > 0.0 else 0.0)
            keypoint_rows.append(kpts)
            visibility_rows.append(vis)

        if not bbox_rows:
            return _empty_targets()
        return (
            torch.tensor(bbox_rows, dtype=torch.float32),
            torch.tensor(keypoint_rows, dtype=torch.float32),
            torch.tensor(visibility_rows, dtype=torch.float32),
        )


def _empty_targets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(0, 4, dtype=torch.float32),
        torch.zeros(0, N_KEYPOINTS, 2, dtype=torch.float32),
        torch.zeros(0, N_KEYPOINTS, dtype=torch.float32),
    )


def collate_pose_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    images, bboxes, keypoints, visibility = zip(*batch)
    return torch.stack(list(images)), list(bboxes), list(keypoints), list(visibility)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV2-skipless wheel pose detector"
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Synthetic default is 4; for real data this optionally caps train steps.",
    )
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
        "--dataset-root",
        type=Path,
        default=None,
        help="YOLO-pose dataset root with images/{train,val} and labels/{train,val}.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialise encoder from torchvision ImageNet weights (needs internet).",
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Path to an existing .pt checkpoint (last.pt) to warm-start from. "
        "Loads after --pretrained init; raises on shape mismatch.",
    )
    return p.parse_args(argv)


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


def _match_to_device(match: MatchResult, device: torch.device) -> MatchResult:
    return MatchResult(
        cls_target=match.cls_target.to(device),
        bbox_target=match.bbox_target.to(device),
        kpt_target=match.kpt_target.to(device),
        vis_target=match.vis_target.to(device),
        pos_mask=match.pos_mask.to(device),
    )


def _loss_line(
    prefix: str,
    loss_dict: dict[str, torch.Tensor],
    lr: float | None = None,
) -> str:
    parts = [
        prefix,
        f"total={float(loss_dict['total'].detach()):.4f}",
        f"cls={float(loss_dict['cls']):.4f}",
        f"bbox={float(loss_dict['bbox']):.4f}",
        f"kpt={float(loss_dict['kpt']):.4f}",
        f"vis={float(loss_dict['vis']):.4f}",
    ]
    if lr is not None:
        parts.append(f"lr={lr:.6f}")
    return " ".join(parts)


def _run_batch(
    model: MobileNetV2SkiplessPose,
    criterion: PoseLoss,
    images: torch.Tensor,
    gt_bb: list[torch.Tensor],
    gt_kp: list[torch.Tensor],
    gt_v: list[torch.Tensor],
    grid_hw: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = images.to(device)
    preds = model(images)
    # Matcher is tiny and CPU-side avoids MPS crashes around scalar indexing.
    match = assign_targets_batched(gt_bb, gt_kp, gt_v, grid_hw, FEATURE_STRIDE)
    match = _match_to_device(match, device)
    return criterion(preds, match)


def _iter_train_loader(
    train_loader: DataLoader,
    max_steps: int | None,
):
    for step_idx, batch in enumerate(train_loader, start=1):
        if max_steps is not None and step_idx > max_steps:
            break
        yield step_idx, batch


def _average_loss(total_loss: float, steps: int) -> float:
    return total_loss / max(steps, 1)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"non-finite final loss: {value}")
    return value


def _json_ready_config(
    args: argparse.Namespace, imgsz: int, grid_hw: tuple[int, int]
) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["imgsz"] = imgsz
    config["grid_hw"] = list(grid_hw)
    return config


def _dataset_summary(dataset: YoloPoseDataset | None) -> dict | None:
    if dataset is None:
        return None
    return {
        "image_count": dataset.stats.image_count,
        "labelled_wheel_count": dataset.stats.labelled_wheel_count,
        "empty_label_count": dataset.stats.empty_label_count,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if args.imgsz % FEATURE_STRIDE != 0:
        raise ValueError(f"--imgsz must be divisible by {FEATURE_STRIDE}")

    out_dir = args.project / args.name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = MobileNetV2SkiplessPose(pretrained=args.pretrained).to(device)
    if args.init_from is not None:
        ckpt_path = Path(args.init_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for k in ("model_state_dict", "model", "state_dict"):
                if k in state:
                    state = state[k]
                    break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"--init-from: missing keys (init from scratch): {len(missing)}")
        if unexpected:
            print(f"--init-from: unexpected keys (ignored): {len(unexpected)}")
        print(f"--init-from: loaded {ckpt_path}")
    criterion = PoseLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    real_data = args.dataset_root is not None
    train_loader = None
    val_loader = None
    train_dataset = None
    val_dataset = None
    if real_data:
        train_dataset = YoloPoseDataset(
            args.dataset_root, "train", imgsz=args.imgsz, limit=args.limit_train
        )
        val_dataset = YoloPoseDataset(
            args.dataset_root, "val", imgsz=args.imgsz, limit=args.limit_val
        )
        if len(train_dataset) == 0:
            raise ValueError(f"no train images found under {args.dataset_root}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_pose_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_pose_batch,
        )
        steps_per_epoch = min(
            len(train_loader), args.steps_per_epoch or len(train_loader)
        )
        val_steps = len(val_loader)
    else:
        steps_per_epoch = args.steps_per_epoch or 4
        val_steps = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * steps_per_epoch, 1), eta_min=1e-5
    )

    imgsz = args.imgsz
    grid_hw = (imgsz // FEATURE_STRIDE, imgsz // FEATURE_STRIDE)

    log_lines: list[str] = []
    print(f"device:         {device}")
    print(f"mode:           {'real YOLO-pose' if real_data else 'synthetic smoke'}")
    print(f"epochs × steps: {args.epochs} × {steps_per_epoch}")
    print(f"batch size:     {args.batch}")
    print(f"image size:     {imgsz}")
    print(f"out dir:        {out_dir}")
    print(f"pretrained enc: {args.pretrained}")
    if real_data:
        print(f"dataset root:   {args.dataset_root}")
        print(f"train images:   {train_dataset.stats.image_count}")  # type: ignore[union-attr]
        print(f"val images:     {val_dataset.stats.image_count}")  # type: ignore[union-attr]
    print()

    started_at = time.time()
    global_step = 0
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_steps = 0
        if real_data:
            assert train_loader is not None
            train_batches = _iter_train_loader(train_loader, args.steps_per_epoch)
        else:
            train_batches = (
                (step + 1, _synthetic_batch(args.batch, imgsz))
                for step in range(steps_per_epoch)
            )

        for step, batch in train_batches:
            images, gt_bb, gt_kp, gt_v = batch
            ld = _run_batch(
                model, criterion, images, gt_bb, gt_kp, gt_v, grid_hw, device
            )

            optimizer.zero_grad()
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            seen_steps += 1
            epoch_loss += float(ld["total"].detach())

            line = _loss_line(
                f"epoch {epoch}/{args.epochs} step {step}/{steps_per_epoch}",
                ld,
                scheduler.get_last_lr()[0],
            )
            print(line)
            log_lines.append(line)

        final_train_loss = _average_loss(epoch_loss, seen_steps)
        avg_line = f"epoch {epoch} train_avg_total={final_train_loss:.4f}"
        print(f"  {avg_line}")
        log_lines.append(avg_line)

        if real_data and val_loader is not None and val_steps > 0:
            model.train(False)
            val_loss = 0.0
            val_seen = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images, gt_bb, gt_kp, gt_v = batch
                    ld = _run_batch(
                        model, criterion, images, gt_bb, gt_kp, gt_v, grid_hw, device
                    )
                    val_loss += float(ld["total"].detach())
                    val_seen += 1
                    line = _loss_line(
                        f"epoch {epoch}/{args.epochs} val {val_step}/{val_steps}", ld
                    )
                    print(line)
                    log_lines.append(line)
            final_val_loss = _average_loss(val_loss, val_seen)
            val_line = f"epoch {epoch} val_avg_total={final_val_loss:.4f}"
            print(f"  {val_line}")
            log_lines.append(val_line)

    duration = time.time() - started_at
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "config": _json_ready_config(args, imgsz, grid_hw),
    }
    ckpt_path = weights_dir / "last.pt"
    torch.save(ckpt, ckpt_path)
    (out_dir / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    summary = {
        "mode": "real_yolo_pose" if real_data else "synthetic_smoke",
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "train": _dataset_summary(train_dataset),
        "val": _dataset_summary(val_dataset),
        "epochs": args.epochs,
        "batch": args.batch,
        "device": str(device),
        "pretrained": args.pretrained,
        "imgsz": imgsz,
        "num_workers": args.num_workers,
        "global_step": global_step,
        "duration_seconds": duration,
        "final_losses": {
            "train_total": _finite_or_none(final_train_loss),
            "val_total": _finite_or_none(final_val_loss),
        },
        "checkpoint": str(ckpt_path),
        "provisional_data_note": (
            "0003 remains provisional until the data-quality gate passes"
            if real_data
            else "synthetic smoke only; not a usable model"
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nduration: {duration:.1f}s")
    print(f"checkpoint: {ckpt_path}")
    print(f"summary:    {out_dir / 'run_summary.json'}")
    if real_data:
        print("NOTE: this is a provisional baseline. 0003 is not production-approved.")
    else:
        print("NOTE: this trained on SYNTHETIC random data. Not a usable model.")


if __name__ == "__main__":
    main()
