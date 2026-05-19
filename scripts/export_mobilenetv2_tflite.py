"""Convert MobileNetV2 ONNX artifacts to TFLite/LiteRT with guarded reports.

The main VSBL venv intentionally does not install TensorFlow. This script is
therefore built to run the converter through a separate Python executable
(`--converter-python`) and to write an explicit BLOCKED_* report when the
converter/runtime dependencies are missing.

Expected flow once a clean converter env exists:

    python3 -m venv .tflite-venv
    .tflite-venv/bin/pip install --upgrade pip
    .tflite-venv/bin/pip install tensorflow onnx2tf
    ./.venv/bin/python scripts/export_mobilenetv2_tflite.py \
      --onnx-path outputs/.../model.onnx \
      --sample-image outputs/.../sample.jpg \
      --converter-python .tflite-venv/bin/python \
      --out-dir outputs/.../tflite \
      --name model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from export_mobilenetv2_skipless import (  # noqa: E402
    decoded_parity_report,
    export_onnx,
    load_model,
    load_sample_tensor,
    onnx_raw_outputs,
    raw_parity_report,
)


OUTPUT_NAMES = ("cls", "bbox", "kpt", "vis")
DEFAULT_MODEL_STATUS = "provisional_0003_not_production"
REQUIRED_CONVERTER_MODULES = (
    "tensorflow",
    "onnx2tf",
    "onnx",
    "onnx_graphsurgeon",
    "sng4onnx",
    "onnxsim",
    "psutil",
    "tf_keras",
)


def converter_subprocess_env(python: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = python.parent if python.parent != Path("") else Path(sys.executable).parent
    if not python_bin.is_absolute():
        python_bin = (REPO / python_bin).resolve()
    env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MobileNetV2 ONNX export to TFLite/LiteRT"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx-path", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--converter-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable from the isolated TFLite converter env.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--bbox-atol", type=float, default=3.0)
    parser.add_argument("--kpt-atol", type=float, default=4.0)
    parser.add_argument("--conf-atol", type=float, default=0.06)
    parser.add_argument("--raw-atol", type=float, default=2e-3)
    parser.add_argument("--keep-saved-model", action="store_true")
    return parser.parse_args(argv)


def run_python(
    python: Path,
    code: str,
    *,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), "-c", code, *(args or [])],
        cwd=REPO,
        env=converter_subprocess_env(python),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def module_available(python: Path, module: str) -> bool:
    proc = run_python(
        python,
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
        args=[module],
    )
    return proc.returncode == 0


def command_available_in_python_env(python: Path, command: str) -> bool:
    return command_path_in_python_env(python, command) is not None


def command_path_in_python_env(python: Path, command: str) -> str | None:
    proc = run_python(
        python,
        "import shutil, sys; path = shutil.which(sys.argv[1]); print(path or ''); sys.exit(0 if path else 1)",
        args=[command],
    )
    path = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return path if proc.returncode == 0 and path else None


def setup_command(converter_python: Path) -> str:
    return "\n".join(
        [
            "python3 -m venv .tflite-venv",
            ".tflite-venv/bin/pip install --upgrade pip",
            ".tflite-venv/bin/pip install tensorflow onnx2tf onnx onnxruntime "
            "onnxsim onnx-graphsurgeon sng4onnx psutil tf_keras",
            "./.venv/bin/python scripts/export_mobilenetv2_tflite.py \\",
            "  --onnx-path outputs/mobilenetv2_export/mn2_0003_kpt_smoothl1_e20_onnx/mn2_0003_kpt_smoothl1_e20.onnx \\",
            "  --sample-image outputs/unreal_export_acceptance/unreal_0003/pose_dataset/images/val/unreal_0003__1004.jpg \\",
            "  --converter-python .tflite-venv/bin/python \\",
            "  --out-dir outputs/mobilenetv2_export/mn2_0003_kpt_smoothl1_e20_tflite \\",
            "  --name mn2_0003_kpt_smoothl1_e20",
        ]
    )


def dependency_status(converter_python: Path) -> dict[str, Any]:
    python_exists = (
        converter_python.is_file() or shutil.which(str(converter_python)) is not None
    )
    modules = {
        module: python_exists and module_available(converter_python, module)
        for module in REQUIRED_CONVERTER_MODULES
    }
    tensorflow = modules["tensorflow"]
    onnx2tf_module = python_exists and module_available(converter_python, "onnx2tf")
    onnx2tf_command_path = (
        command_path_in_python_env(converter_python, "onnx2tf") if python_exists else None
    )
    missing_modules = [
        module for module, available in modules.items() if not available
    ]
    return {
        "converter_python": str(converter_python),
        "python_exists": python_exists,
        "tensorflow": tensorflow,
        "onnx2tf_module": onnx2tf_module,
        "onnx2tf_command": onnx2tf_command_path is not None,
        "onnx2tf_command_path": onnx2tf_command_path,
        "converter_modules": modules,
        "missing_modules": missing_modules,
        "ready": (
            python_exists
            and not missing_modules
            and (onnx2tf_module or onnx2tf_command_path)
        ),
        "setup_command": setup_command(converter_python),
    }


def _blocked_status(deps: dict[str, Any]) -> str | None:
    if not deps["python_exists"]:
        return "BLOCKED_MISSING_CONVERTER_PYTHON"
    if not deps["tensorflow"]:
        return "BLOCKED_MISSING_TENSORFLOW"
    if not (deps["onnx2tf_module"] or deps["onnx2tf_command"]):
        return "BLOCKED_MISSING_CONVERTER"
    if deps.get("missing_modules"):
        return "BLOCKED_MISSING_CONVERTER_DEPENDENCIES"
    return None


def ensure_onnx(args: argparse.Namespace) -> Path:
    if args.onnx_path is not None:
        if not args.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {args.onnx_path}")
        return args.onnx_path
    if args.checkpoint is None:
        raise ValueError("either --onnx-path or --checkpoint is required")
    device = torch.device("cpu")
    model = load_model(args.checkpoint, device=device)
    onnx_path = args.out_dir / f"{args.name}.onnx"
    export_onnx(model, onnx_path, imgsz=args.imgsz, opset=args.opset, device=device)
    return onnx_path


def convert_onnx_to_saved_model(
    *,
    converter_python: Path,
    onnx_path: Path,
    saved_model_dir: Path,
    input_name: str,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
    if module_available(converter_python, "onnx2tf"):
        command = [
            str(converter_python),
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(saved_model_dir),
            "-nuo",
            "-dsm",
            "-k",
            input_name,
        ]
    else:
        command_path = command_path_in_python_env(converter_python, "onnx2tf")
        if command_path is None:
            return subprocess.CompletedProcess([], 127, "onnx2tf command not found\n")
        command = [
            command_path,
            "-i",
            str(onnx_path),
            "-o",
            str(saved_model_dir),
            "-nuo",
            "-dsm",
            "-k",
            input_name,
        ]
    proc = subprocess.run(
        command,
        cwd=REPO,
        env=converter_subprocess_env(converter_python),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def convert_saved_model_to_tflite(
    *,
    converter_python: Path,
    saved_model_dir: Path,
    tflite_path: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    code = r"""
import pathlib
import sys
import tensorflow as tf

saved_model_dir = pathlib.Path(sys.argv[1])
tflite_path = pathlib.Path(sys.argv[2])
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_path.parent.mkdir(parents=True, exist_ok=True)
tflite_path.write_bytes(tflite_model)
print(f"wrote {tflite_path} bytes={len(tflite_model)}")
"""
    proc = run_python(
        converter_python,
        code,
        args=[str(saved_model_dir), str(tflite_path)],
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def find_onnx2tf_tflite(saved_model_dir: Path, name: str) -> Path | None:
    candidates = [
        saved_model_dir / f"{name}_float32.tflite",
        *sorted(saved_model_dir.glob("*_float32.tflite")),
        *sorted(saved_model_dir.glob("*.tflite")),
    ]
    for candidate in candidates:
        if candidate.is_file() and "float16" not in candidate.name:
            return candidate
    return None


def tflite_raw_outputs(
    *,
    converter_python: Path,
    tflite_path: Path,
    sample_tensor_path: Path,
    output_npz_path: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    code = r"""
import numpy as np
import pathlib
import sys
import tensorflow as tf

tflite_path = pathlib.Path(sys.argv[1])
sample_tensor_path = pathlib.Path(sys.argv[2])
output_npz_path = pathlib.Path(sys.argv[3])

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sample = np.load(sample_tensor_path).astype(np.float32)
input_detail = input_details[0]
input_shape = tuple(int(v) for v in input_detail["shape"])
if tuple(sample.shape) == input_shape:
    input_value = sample
elif sample.ndim == 4 and tuple(sample.transpose(0, 2, 3, 1).shape) == input_shape:
    input_value = sample.transpose(0, 2, 3, 1)
else:
    raise ValueError(f"unsupported TFLite input shape {input_shape} for sample {sample.shape}")
if not np.issubdtype(input_detail["dtype"], np.floating):
    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale:
        input_value = np.round(input_value / scale + zero_point)
    input_value = np.clip(
        input_value,
        np.iinfo(input_detail["dtype"]).min,
        np.iinfo(input_detail["dtype"]).max,
    ).astype(input_detail["dtype"])
else:
    input_value = input_value.astype(input_detail["dtype"])
interpreter.set_tensor(input_detail["index"], input_value)
interpreter.invoke()
outputs = {}
meta = {}
for i, detail in enumerate(output_details):
    value = interpreter.get_tensor(detail["index"])
    q = detail.get("quantization_parameters") or {}
    scales = q.get("scales")
    zero_points = q.get("zero_points")
    if not np.issubdtype(value.dtype, np.floating) and scales is not None and len(scales):
        scales = np.asarray(scales, dtype=np.float32)
        zero_points = np.asarray(zero_points, dtype=np.float32)
        if scales.size == 1:
            value = (value.astype(np.float32) - float(zero_points[0])) * float(scales[0])
        else:
            axis = int(q.get("quantized_dimension", -1))
            if axis < 0:
                axis += value.ndim
            shape = [1] * value.ndim
            shape[axis] = scales.size
            value = (value.astype(np.float32) - zero_points.reshape(shape)) * scales.reshape(shape)
    else:
        value = value.astype(np.float32)
    outputs[f"output_{i}"] = value
    meta[f"output_{i}_shape"] = np.asarray(value.shape, dtype=np.int64)
output_npz_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(output_npz_path, **outputs, **meta)
print(f"wrote {output_npz_path} outputs={len(output_details)}")
"""
    proc = run_python(
        converter_python,
        code,
        args=[str(tflite_path), str(sample_tensor_path), str(output_npz_path)],
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def normalize_tflite_outputs(
    npz_path: Path,
    expected_shapes: dict[str, list[int]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loaded = np.load(npz_path)
    arrays = {
        key: loaded[key].astype(np.float32)
        for key in loaded.files
        if key.startswith("output_") and not key.endswith("_shape")
    }
    mapped: dict[str, np.ndarray] = {}
    used: set[str] = set()
    failures: list[str] = []
    observed_shapes = {key: list(value.shape) for key, value in arrays.items()}

    for name in OUTPUT_NAMES:
        expected = tuple(expected_shapes[name])
        found_key = None
        found_value = None
        for key, value in arrays.items():
            if key in used:
                continue
            if tuple(value.shape) == expected:
                found_key = key
                found_value = value
                break
            if value.ndim == 4:
                transposed = np.transpose(value, (0, 3, 1, 2))
                if tuple(transposed.shape) == expected:
                    found_key = key
                    found_value = transposed
                    break
        if found_key is None or found_value is None:
            failures.append(f"could not map TFLite output for {name} expected {expected}")
            continue
        used.add(found_key)
        mapped[name] = found_value

    return (
        {
            "mapped": not failures,
            "failures": failures,
            "observed_shapes": observed_shapes,
            "mapped_shapes": {key: list(value.shape) for key, value in mapped.items()},
        },
        mapped,
    )


def _np_to_torch(raw: dict[str, np.ndarray]):
    import torch

    return {name: torch.from_numpy(value) for name, value in raw.items()}


def _write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tflite_export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MobileNetV2 TFLite/LiteRT Export Report",
        "",
        f"- Status: **{report['status']}**",
        f"- ONNX: `{report.get('onnx_path', '')}`",
        f"- TFLite: `{report.get('tflite_path', '')}`",
        f"- Converter python: `{report['dependencies']['converter_python']}`",
        f"- Model status: **{report['model_status']}**",
        "",
    ]
    if report["status"].startswith("BLOCKED"):
        lines += [
            "## Setup",
            "",
            "Converter/runtime dependencies are intentionally expected in a separate env:",
            "",
            "```bash",
            report["dependencies"]["setup_command"],
            "```",
            "",
        ]
    if report.get("raw_parity"):
        lines += [
            "## Parity",
            "",
            f"- Raw parity: **{report['raw_parity']['matched']}** "
            f"(max abs diff {report['raw_parity']['max_abs_diff']:.6g})",
            f"- Decoded parity: **{report['decoded_parity']['matched']}**",
            f"- PyTorch detections: {report['decoded_parity']['n_pytorch']}",
            f"- TFLite detections: {report['decoded_parity']['n_onnx']}",
            "",
        ]
    if report.get("failures"):
        lines += ["## Failures", ""]
        lines.extend(f"- {failure}" for failure in report["failures"])
        lines.append("")
    if report.get("warnings"):
        lines += ["## Warnings", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    lines += [
        "Note: this export remains provisional until clean-export retraining and "
        "Android device validation are complete.",
        "",
    ]
    (out_dir / "tflite_export_report.md").write_text("\n".join(lines), encoding="utf-8")


def _logs_report(logs_dir: Path) -> dict[str, str]:
    candidates = {
        "onnx2tf": logs_dir / "01_onnx2tf.log",
        "saved_model_to_tflite": logs_dir / "02_saved_model_to_tflite.log",
        "tflite_runtime": logs_dir / "03_tflite_runtime.log",
    }
    return {name: str(path) for name, path in candidates.items() if path.is_file()}


def _base_report(
    args: argparse.Namespace, deps: dict[str, Any], onnx_path: Path | None
) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "onnx_path": str(onnx_path) if onnx_path else None,
        "sample_image": str(args.sample_image),
        "tflite_path": str(args.out_dir / f"{args.name}.tflite"),
        "saved_model_dir": str(args.out_dir / f"{args.name}_saved_model"),
        "imgsz": args.imgsz,
        "input_name": args.input_name,
        "onnx2tf_options": {
            "not_use_onnxsim": True,
            "disable_strict_mode": True,
            "keep_nchw_input": args.input_name,
        },
        "thresholds": {
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "bbox_atol": args.bbox_atol,
            "kpt_atol": args.kpt_atol,
            "conf_atol": args.conf_atol,
            "raw_atol": args.raw_atol,
        },
        "dependencies": deps,
        "model_status": DEFAULT_MODEL_STATUS,
        "failures": [],
        "warnings": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deps = dependency_status(args.converter_python)
    onnx_path = ensure_onnx(args)
    report = _base_report(args, deps, onnx_path)
    blocked = _blocked_status(deps)
    if blocked is not None:
        report["status"] = blocked
        report["failures"] = [f"{blocked}: converter dependencies are not ready"]
        _write_report(report, args.out_dir)
        return report

    saved_model_dir = args.out_dir / f"{args.name}_saved_model"
    tflite_path = args.out_dir / f"{args.name}.tflite"
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for stale_log in logs_dir.glob("*.log"):
        stale_log.unlink()

    onnx_to_tf = convert_onnx_to_saved_model(
        converter_python=args.converter_python,
        onnx_path=onnx_path,
        saved_model_dir=saved_model_dir,
        input_name=args.input_name,
        log_path=logs_dir / "01_onnx2tf.log",
    )
    if not (saved_model_dir / "saved_model.pb").is_file():
        report["status"] = "CONVERSION_FAILED"
        report["failures"] = [
            f"ONNX to SavedModel conversion failed with exit code {onnx_to_tf.returncode}"
        ]
        report["logs"] = _logs_report(logs_dir)
        _write_report(report, args.out_dir)
        return report
    if onnx_to_tf.returncode != 0:
        report["warnings"].append(
            "onnx2tf returned a non-zero exit code but saved_model.pb exists; "
            "continuing to TFLite runtime parity"
        )

    onnx2tf_tflite = find_onnx2tf_tflite(saved_model_dir, args.name)
    source_tflite_path = None
    if onnx2tf_tflite is not None:
        shutil.copy2(onnx2tf_tflite, tflite_path)
        source_tflite_path = onnx2tf_tflite
        report["warnings"].append(
            f"Using onnx2tf-generated TFLite artifact: {onnx2tf_tflite}"
        )
    else:
        tf_to_tflite = convert_saved_model_to_tflite(
            converter_python=args.converter_python,
            saved_model_dir=saved_model_dir,
            tflite_path=tflite_path,
            log_path=logs_dir / "02_saved_model_to_tflite.log",
        )
        if tf_to_tflite.returncode != 0 or not tflite_path.is_file():
            report["status"] = "CONVERSION_FAILED"
            report["failures"] = ["SavedModel to TFLite conversion failed"]
            report["logs"] = _logs_report(logs_dir)
            _write_report(report, args.out_dir)
            return report

    image_tensor, _ = load_sample_tensor(args.sample_image, args.imgsz)
    npy_path = args.out_dir / "sample_input.npy"
    np.save(npy_path, image_tensor.unsqueeze(0).numpy())
    tflite_npz = args.out_dir / "tflite_outputs.npz"
    runtime = tflite_raw_outputs(
        converter_python=args.converter_python,
        tflite_path=tflite_path,
        sample_tensor_path=npy_path,
        output_npz_path=tflite_npz,
        log_path=logs_dir / "03_tflite_runtime.log",
    )
    if runtime.returncode != 0 or not tflite_npz.is_file():
        report["status"] = "RUNTIME_FAILED"
        report["failures"] = ["TFLite runtime invocation failed"]
        report["logs"] = _logs_report(logs_dir)
        _write_report(report, args.out_dir)
        return report

    onnx_raw = onnx_raw_outputs(onnx_path, image_tensor)
    expected_shapes = {name: list(value.shape) for name, value in onnx_raw.items()}
    mapping_report, tflite_np = normalize_tflite_outputs(tflite_npz, expected_shapes)
    if not mapping_report["mapped"]:
        report["status"] = "PARITY_FAILED"
        report["failures"] = mapping_report["failures"]
        report["tflite_output_mapping"] = mapping_report
        _write_report(report, args.out_dir)
        return report

    tflite_raw = _np_to_torch(tflite_np)
    raw_parity = raw_parity_report(onnx_raw, tflite_raw, raw_atol=args.raw_atol)
    decoded_parity, _ = decoded_parity_report(
        onnx_raw,
        tflite_raw,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        bbox_atol=args.bbox_atol,
        kpt_atol=args.kpt_atol,
        conf_atol=args.conf_atol,
    )
    report.update(
        {
            "status": (
                "PASS"
                if raw_parity["matched"] and decoded_parity["matched"]
                else "PARITY_FAILED"
            ),
            "tflite_output_mapping": mapping_report,
            "source_tflite_path": (
                str(source_tflite_path) if source_tflite_path is not None else None
            ),
            "raw_parity": raw_parity,
            "decoded_parity": decoded_parity,
            "logs": _logs_report(logs_dir),
        }
    )
    report["failures"] = raw_parity["failures"] + decoded_parity["failures"]
    if not args.keep_saved_model:
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    _write_report(report, args.out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(f"Status:        {report['status']}")
    print(f"Report:        {args.out_dir / 'tflite_export_report.md'}")
    print(f"ONNX:          {report.get('onnx_path')}")
    print(f"TFLite:        {report.get('tflite_path')}")
    if report["status"].startswith("BLOCKED"):
        print("Setup command:")
        print(report["dependencies"]["setup_command"])
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
