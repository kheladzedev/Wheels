#!/usr/bin/env python3
"""Export the web floor multi-task model to ONNX and write handoff files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from web_floor_export import (
    decoded_sample_from_onnx,
    export_web_floor_onnx,
    onnxruntime_shape_smoke,
    write_handoff_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/pose_dataset_web_floor_fixture.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/web_floor_network/handoff"))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "web_floor_multitask.onnx"
    export_web_floor_onnx(
        checkpoint=args.checkpoint,
        onnx_path=onnx_path,
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
    )
    smoke = onnxruntime_shape_smoke(onnx_path, imgsz=args.imgsz)
    (args.out_dir / "python_onnx_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    decoded = decoded_sample_from_onnx(
        onnx_path=onnx_path,
        config=args.config,
        sample_index=args.sample_index,
        imgsz=args.imgsz,
    )
    (args.out_dir / "sample_decoded.json").write_text(json.dumps(decoded, indent=2), encoding="utf-8")
    manifest = write_handoff_manifest(
        manifest_path=args.out_dir / "manifest.json",
        onnx_path=onnx_path,
        smoke=smoke,
        distance_mode=decoded["floor"]["distance_mode"],
    )
    web_readme = ROOT / "web_handoff" / "README.md"
    if web_readme.is_file():
        shutil.copy2(web_readme, args.out_dir / "README.md")
    print(json.dumps({
        "onnx_path": str(onnx_path),
        "manifest": str(args.out_dir / "manifest.json"),
        "python_smoke_ok": smoke["ok"],
        "input_shape": smoke["input_shape"],
        "output_shapes": manifest["output_shapes"],
        "runtime_scope": manifest["runtime_scope"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
