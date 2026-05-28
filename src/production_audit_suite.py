"""Run the production-readiness audit suite in a safe, repeatable order.

The individual audit tools are intentionally small and focused. This
runner handles orchestration so the handoff reports, gates, release
hashes, and senior evidence matrix are regenerated consistently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .release_integrity import required_artifacts_for_current_state
except ImportError:  # pragma: no cover - used when executed as a script
    from release_integrity import required_artifacts_for_current_state


DEFAULT_STATUS_OUT = Path("outputs/production_audit/audit_suite_status.json")
DEFAULT_EVIDENCE_PREFLIGHT_STATUS_OUT = Path(
    "outputs/production_audit/production_evidence_intake_preflight_status.json"
)
OBJECTIVE_COMPLETION_ARTIFACTS = {
    "src/objective_completion_audit.py",
    "docs/OBJECTIVE_COMPLETION_AUDIT.md",
    "outputs/production_audit/objective_completion_audit.json",
}


@dataclass
class Step:
    name: str
    cmd: list[str]
    allow_failure: bool = False


@dataclass
class StepResult:
    name: str
    returncode: int
    allowed_failure: bool
    ok: bool


def py_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def release_integrity_cmd(*, include_objective: bool) -> list[str]:
    artifacts = required_artifacts_for_current_state(include_objective=include_objective)
    cmd = py_cmd("src/release_integrity.py")
    for artifact in artifacts:
        cmd.extend(["--artifact", artifact])
    return cmd


def build_steps(*, include_performance: bool, include_pytest: bool) -> list[Step]:
    steps = [
        Step("model_inventory", py_cmd("src/model_inventory.py")),
        Step("model_selection_audit", py_cmd("src/model_selection_audit.py")),
        Step("dataset_audit", py_cmd("src/dataset_audit.py")),
        Step("runtime_contract_audit", py_cmd("src/runtime_contract_audit.py")),
        Step("spec_compliance_audit", py_cmd("src/spec_compliance_audit.py")),
    ]
    if include_performance:
        steps.append(Step("performance_audit", py_cmd("src/performance_audit.py")))
    steps.extend(
        [
            Step("export_parity_audit", py_cmd("src/export_parity_audit.py")),
            Step("export_certification", py_cmd("src/export_certification.py")),
            Step("tflite_certification", py_cmd("src/tflite_certification.py")),
            Step(
                "android_litert_report_template",
                py_cmd("scripts/create_android_litert_report_template.py"),
            ),
            Step(
                "ar_replay_log_template",
                py_cmd("scripts/create_ar_replay_log_template.py"),
            ),
            Step(
                "ar_holdout_provenance_template",
                py_cmd("scripts/create_ar_holdout_provenance_template.py"),
            ),
            Step(
                "production_evidence_intake_preflight",
                py_cmd(
                    "src/run_production_evidence_intake.py",
                    "--dry-run",
                    "--status-out",
                    str(DEFAULT_EVIDENCE_PREFLIGHT_STATUS_OUT),
                ),
                allow_failure=True,
            ),
            Step(
                "external_evidence_return_template",
                py_cmd("scripts/create_external_evidence_return_template.py"),
            ),
            Step(
                "external_evidence_handoff_bundle",
                py_cmd("scripts/build_external_evidence_handoff_bundle.py"),
            ),
            Step(
                "external_evidence_handoff_bundle_verify",
                py_cmd("src/verify_external_evidence_handoff_bundle.py"),
            ),
            Step("production_evidence_audit", py_cmd("src/production_evidence_audit.py")),
            Step("model_card", py_cmd("src/model_card.py")),
            Step("requirements_traceability", py_cmd("src/requirements_traceability.py")),
            Step("executive_report_ru", py_cmd("src/executive_report_ru.py")),
            # The integration gate itself requires release_integrity to be OK,
            # so generate a pre-gate manifest before gates and a final one
            # after gates/senior audit.
            Step("release_integrity_pregate", release_integrity_cmd(include_objective=False)),
            Step(
                "integration_gate",
                py_cmd(
                    "src/production_gate.py",
                    "--mode",
                    "integration",
                    "--json-out",
                    "outputs/production_audit/integration_gate.json",
                ),
            ),
            Step(
                "production_gate_expected",
                py_cmd(
                    "src/production_gate.py",
                    "--mode",
                    "production",
                    "--json-out",
                    "outputs/production_audit/production_gate.json",
                ),
                allow_failure=True,
            ),
            Step("senior_ml_audit", py_cmd("src/senior_ml_audit.py")),
            Step("requirements_traceability_final", py_cmd("src/requirements_traceability.py")),
            Step("executive_report_ru_final", py_cmd("src/executive_report_ru.py")),
            Step("objective_completion_audit", py_cmd("src/objective_completion_audit.py")),
            Step("release_integrity_final", release_integrity_cmd(include_objective=True)),
            Step("production_readiness_report", py_cmd("scripts/write_production_audit_report.py")),
            Step("handoff_report", py_cmd("scripts/write_handoff_report.py")),
            Step("release_integrity_post_reports", release_integrity_cmd(include_objective=True)),
            Step("report_consistency_audit", py_cmd("src/report_consistency_audit.py")),
            Step("project_readiness", py_cmd("src/project_readiness.py")),
        ]
    )
    if include_pytest:
        steps.append(Step("pytest", [sys.executable, "-m", "pytest"]))
    return steps


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def run_step(step: Step) -> StepResult:
    print(f"\n[audit-suite] step={step.name}")
    print("[audit-suite] cmd=" + " ".join(step.cmd))
    completed = subprocess.run(step.cmd, check=False)
    ok = completed.returncode == 0 or step.allow_failure
    print(
        f"[audit-suite] step={step.name} returncode={completed.returncode} "
        f"allowed_failure={step.allow_failure} ok={ok}"
    )
    return StepResult(
        name=step.name,
        returncode=completed.returncode,
        allowed_failure=step.allow_failure,
        ok=ok,
    )


def evaluate_suite_status(results: list[StepResult], *, strict_production: bool) -> dict[str, Any]:
    integration_gate = read_json(Path("outputs/production_audit/integration_gate.json"))
    production_gate = read_json(Path("outputs/production_audit/production_gate.json"))
    senior_audit = read_json(Path("outputs/production_audit/senior_ml_audit.json"))
    release_integrity = read_json(Path("outputs/production_audit/release_integrity.json"))
    report_consistency = read_json(Path("outputs/production_audit/report_consistency_audit.json"))

    command_failures = [result.name for result in results if not result.ok]
    integration_ready = bool(integration_gate.get("ok")) and bool(
        senior_audit.get("integration_ready")
    )
    report_consistency_ok = bool(report_consistency.get("ok")) and report_consistency.get("failures", []) == []
    production_ready = (
        bool(production_gate.get("ok"))
        and bool(senior_audit.get("production_ready"))
        and report_consistency_ok
    )
    ok = (
        not command_failures
        and integration_ready
        and bool(release_integrity.get("ok"))
        and report_consistency_ok
        and (production_ready if strict_production else True)
    )
    return {
        "ok": ok,
        "strict_production": strict_production,
        "command_failures": command_failures,
        "integration_ready": integration_ready,
        "production_ready": production_ready,
        "production_blockers": senior_audit.get("production_blockers", []),
        "release_integrity_ok": bool(release_integrity.get("ok")),
        "report_consistency_ok": report_consistency_ok,
        "report_consistency_failures": report_consistency.get("failures", []),
        "release_artifacts": release_integrity.get("artifact_count"),
        "steps": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip the local PT/ONNX latency benchmark and reuse the existing report.",
    )
    parser.add_argument(
        "--with-pytest",
        action="store_true",
        help="Run the full pytest suite as the final step.",
    )
    parser.add_argument(
        "--strict-production",
        action="store_true",
        help="Exit non-zero unless the production gate is also green.",
    )
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = build_steps(
        include_performance=not args.skip_performance,
        include_pytest=args.with_pytest,
    )
    results: list[StepResult] = []
    for step in steps:
        if step.name == "report_consistency_audit":
            status = evaluate_suite_status(results, strict_production=args.strict_production)
            args.status_out.parent.mkdir(parents=True, exist_ok=True)
            args.status_out.write_text(
                json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        result = run_step(step)
        results.append(result)
        if not result.ok:
            break

    status = evaluate_suite_status(results, strict_production=args.strict_production)
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    args.status_out.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "\n[audit-suite] "
        f"ok={status['ok']} integration_ready={status['integration_ready']} "
        f"production_ready={status['production_ready']} "
        f"release_integrity_ok={status['release_integrity_ok']}"
    )
    print(f"[audit-suite] production_blockers={status['production_blockers']}")
    print(f"[audit-suite] status={args.status_out}")
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
