"""Smoke-check a TFLite model with the LiteRT Python interpreter.

This is not a replacement for Android-device validation. It proves the
exact `.tflite` artifact can be loaded by `ai_edge_litert`, accepts the
expected input tensor, runs inference, and emits finite raw model output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def preprocess_image(path: Path, input_shape: list[int]) -> np.ndarray:
    if len(input_shape) != 4:
        raise ValueError(f"expected NHWC input shape, got {input_shape}")
    batch, height, width, channels = input_shape
    if batch != 1 or channels != 3:
        raise ValueError(f"expected shape [1,H,W,3], got {input_shape}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return (image.astype(np.float32) / 255.0)[None, ...]


def _tensor_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "min": float(np.min(value)) if value.size else None,
        "max": float(np.max(value)) if value.size else None,
        "mean": float(np.mean(value)) if value.size else None,
    }


def run_litert_check(model_path: Path, image_path: Path, warmup: int, runs: int) -> dict[str, Any]:
    from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise RuntimeError(f"expected one input tensor, got {len(input_details)}")
    if not output_details:
        raise RuntimeError("model exposes no output tensors")

    input_detail = input_details[0]
    input_shape = [int(v) for v in input_detail["shape"]]
    input_tensor = preprocess_image(image_path, input_shape).astype(input_detail["dtype"])

    input_index = int(input_detail["index"])
    latencies_ms: list[float] = []
    total_runs = max(0, warmup) + max(1, runs)
    outputs: list[np.ndarray] = []
    for i in range(total_runs):
        interpreter.set_tensor(input_index, input_tensor)
        start = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            latencies_ms.append(elapsed_ms)
        outputs = [interpreter.get_tensor(int(detail["index"])) for detail in output_details]

    finite_outputs = all(np.isfinite(output).all() for output in outputs)
    return {
        "ok": finite_outputs,
        "model": str(model_path),
        "image": str(image_path),
        "runtime": "ai_edge_litert",
        "input": {
            "shape": input_shape,
            "dtype": str(input_detail["dtype"]),
        },
        "outputs": [_tensor_summary(output) for output in outputs],
        "latency_ms": {
            "runs": len(latencies_ms),
            "mean": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/production_audit/litert_runtime_smoke.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    report = run_litert_check(args.model, args.image, args.warmup, args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"ok={report['ok']} runtime={report['runtime']} "
        f"output_shapes={[o['shape'] for o in report['outputs']]} "
        f"mean_latency_ms={report['latency_ms']['mean']}"
    )
    print(f"report={args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
