"""Production-readiness gate check.

Inspects on-disk signals and emits a fail-closed verdict on whether
training, AR-ready claims, or production claims are allowed. Run
before stating any "ready" claim:

    ./.venv/bin/python scripts/prod_readiness_check.py

Reads (all optional — missing files are treated as failing gates):

- ``data/incoming/*/metadata/acceptance_status.json``
- ``outputs/full_pipeline_audit/REPORT.json``
- ``outputs/awe_demo/demo_summary.json``
- ``runs/pose/*/SEMANTICS.md``

Writes:

- ``outputs/prod_readiness/REPORT.json``  (machine-readable verdict)
- ``outputs/prod_readiness/REPORT.md``    (human-readable)

Definitions and gate semantics live in
``docs/PRODUCTION_READINESS_PLAN.md``. Per-batch field semantics live
in ``docs/DATASET_ACCEPTANCE_TEMPLATE.md``. Per-model gate signals
live in ``docs/MODEL_CARD_TEMPLATE.md`` (§12).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "outputs" / "prod_readiness"
DEFAULT_DATA_INCOMING_ROOT = ROOT / "data" / "incoming"
DEFAULT_AUDIT_REPORT_PATH = ROOT / "outputs" / "full_pipeline_audit" / "REPORT.json"
DEFAULT_DEMO_SUMMARY_PATH = ROOT / "outputs" / "awe_demo" / "demo_summary.json"
DEFAULT_RUNS_POSE_ROOT = ROOT / "runs" / "pose"

ACCEPTANCE_STATUSES = (
    "ACCEPT_FOR_TRAINING",
    "ACCEPT_ONLY_AS_DEBUG",
    "REJECT_NEEDS_FIX",
)
RESULT_OK = "PASS"
RESULT_NEUTRAL = {"PASS", "WARN", "FAIL", "NOT_RUN"}
FORBIDDEN_BBOX_SOURCES = {"PLACEHOLDER", "NEEDS_FIX", "UNKNOWN", "", None}
PLACEHOLDER_REVIEWERS = {None, "", "FILL_ME", "TODO", "TBD", "unknown", "null"}

OVERALL_AR_READY = "AR_READY_CANDIDATE"
OVERALL_TRAINING_ALLOWED_AR_BLOCKED = "TRAINING_ALLOWED_AR_BLOCKED"
OVERALL_DEMO_READY = "DEMO_READY_PRODUCTION_BLOCKED"
OVERALL_BLOCKED = "BLOCKED_ON_ACCEPTED_REAL_DATA"


@dataclass
class BatchAcceptance:
    """One per-batch acceptance signal, normalised."""

    batch_id: str
    path: str
    declared_status: str | None
    derived_status: str
    failures: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "path": self.path,
            "declared_status": self.declared_status,
            "derived_status": self.derived_status,
            "failures": list(self.failures),
            "requires_plugin_bbox": bool(
                self.payload.get("requires_plugin_bbox", False)
            ),
            "bbox_source": self.payload.get("bbox_source"),
            "keypoint_mapping": self.payload.get("keypoint_mapping"),
            "human_preview_accepted": bool(
                self.payload.get("human_preview_accepted", False)
            ),
            "human_reviewer": self.payload.get("human_reviewer"),
        }


@dataclass
class SemanticsRecord:
    model_dir: str
    path: str
    fields: dict
    failures: list[str] = field(default_factory=list)

    @property
    def is_real_floorray(self) -> bool:
        return (
            self.fields.get("semantics_version") == "floorray_v1"
            and self.fields.get("trained_on_real_data") is True
            and self.fields.get("stale") is False
        )

    def to_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "path": self.path,
            "fields": dict(self.fields),
            "failures": list(self.failures),
            "is_real_floorray": self.is_real_floorray,
        }


def _strtobool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"true", "yes", "1"}:
        return True
    if v in {"false", "no", "0"}:
        return False
    return None


def parse_semantics_md(text: str) -> dict:
    """Parse a ``runs/pose/<name>/SEMANTICS.md`` file (key: value lines).

    Lines without a colon are ignored. Boolean-shaped values are
    normalised to ``True`` / ``False``; other values stay as stripped
    strings. Frontmatter delimiters (``---``) are skipped.
    """
    fields: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        as_bool = _strtobool(value)
        fields[key] = as_bool if as_bool is not None else value
    return fields


def _normalise_result(value: str | None) -> str:
    if value is None:
        return "FAIL"
    value = str(value).strip().upper()
    if value not in RESULT_NEUTRAL:
        return "FAIL"
    return value


def derive_batch_status(payload: dict) -> tuple[str, list[str]]:
    """Return (derived_status, list_of_failure_reasons)."""
    failures: list[str] = []
    if payload.get("schema_version") != 1:
        failures.append("schema_version != 1")
    if bool(payload.get("requires_plugin_bbox", False)):
        failures.append("requires_plugin_bbox is true")
    bbox_source = payload.get("bbox_source")
    if bbox_source in FORBIDDEN_BBOX_SOURCES:
        failures.append(f"bbox_source is placeholder/unknown ({bbox_source!r})")
    if payload.get("keypoint_mapping") != "floorray_v1":
        failures.append(
            f"keypoint_mapping must be floorray_v1, got {payload.get('keypoint_mapping')!r}"
        )
    for field_name in ("validation_result", "preview_result", "bbox_audit_result"):
        if _normalise_result(payload.get(field_name)) != RESULT_OK:
            failures.append(f"{field_name} != PASS")
    if not bool(payload.get("human_preview_accepted", False)):
        failures.append("human_preview_accepted != true")
    reviewer = payload.get("human_reviewer")
    if reviewer in PLACEHOLDER_REVIEWERS or (
        isinstance(reviewer, str) and reviewer.strip() in PLACEHOLDER_REVIEWERS
    ):
        failures.append("human_reviewer missing or placeholder")
    declared = payload.get("status")
    if declared not in ACCEPTANCE_STATUSES:
        failures.append(f"status outside known set ({declared!r})")
    if not failures and declared == "ACCEPT_FOR_TRAINING":
        return "ACCEPT_FOR_TRAINING", failures
    if "requires_plugin_bbox is true" in failures or (
        payload.get("keypoint_mapping") == "rim_v0"
    ):
        return "ACCEPT_ONLY_AS_DEBUG", failures
    if declared == "ACCEPT_ONLY_AS_DEBUG":
        return "ACCEPT_ONLY_AS_DEBUG", failures
    return "REJECT_NEEDS_FIX", failures


def load_acceptance_files(data_incoming_root: Path) -> list[BatchAcceptance]:
    """Load every ``metadata/acceptance_status.json`` under ``data/incoming/``."""
    records: list[BatchAcceptance] = []
    if not data_incoming_root.is_dir():
        return records
    for status_path in sorted(
        data_incoming_root.glob("*/metadata/acceptance_status.json")
    ):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append(
                BatchAcceptance(
                    batch_id=status_path.parent.parent.name,
                    path=str(status_path),
                    declared_status=None,
                    derived_status="REJECT_NEEDS_FIX",
                    failures=[f"could not parse JSON: {exc!s}"],
                )
            )
            continue
        derived, failures = derive_batch_status(payload)
        records.append(
            BatchAcceptance(
                batch_id=str(payload.get("batch_id") or status_path.parent.parent.name),
                path=str(status_path),
                declared_status=payload.get("status"),
                derived_status=derived,
                failures=failures,
                payload=payload,
            )
        )
    return records


def load_audit_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_demo_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_semantics_files(runs_pose_root: Path) -> list[SemanticsRecord]:
    records: list[SemanticsRecord] = []
    if not runs_pose_root.is_dir():
        return records
    for path in sorted(runs_pose_root.glob("*/SEMANTICS.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            records.append(
                SemanticsRecord(
                    model_dir=path.parent.name,
                    path=str(path),
                    fields={},
                    failures=["could not read SEMANTICS.md"],
                )
            )
            continue
        fields = parse_semantics_md(text)
        failures: list[str] = []
        required = ("semantics_version", "trained_on_real_data", "stale", "trained_at")
        for key in required:
            if key not in fields:
                failures.append(f"missing key: {key}")
        if fields.get("semantics_version") not in {"floorray_v1", "rim_v0"}:
            failures.append(
                f"semantics_version must be floorray_v1 or rim_v0, "
                f"got {fields.get('semantics_version')!r}"
            )
        records.append(
            SemanticsRecord(
                model_dir=path.parent.name,
                path=str(path),
                fields=fields,
                failures=failures,
            )
        )
    return records


def _audit_field_pass(audit: dict | None, key: str) -> bool:
    return bool(audit) and audit.get(key) is True


def compute_verdict(
    acceptances: Sequence[BatchAcceptance],
    audit: dict | None,
    demo_summary: dict | None,
    semantics: Sequence[SemanticsRecord],
    *,
    pose_runs_present: bool = False,
    now: datetime | None = None,
) -> dict:
    """Return the machine-readable verdict dict. Pure function."""
    accepted_batches = [
        b for b in acceptances if b.derived_status == "ACCEPT_FOR_TRAINING"
    ]
    debug_batches = [
        b for b in acceptances if b.derived_status == "ACCEPT_ONLY_AS_DEBUG"
    ]
    rejected_batches = [
        b for b in acceptances if b.derived_status == "REJECT_NEEDS_FIX"
    ]

    real_floorray_models = [s for s in semantics if s.is_real_floorray]
    stale_models = [s for s in semantics if s.fields.get("stale") is True]

    geometry_pass = _audit_field_pass(audit, "geometry_audit_pass")
    bbox_pass = _audit_field_pass(audit, "bbox_audit_pass")
    ar_replay_pass = _audit_field_pass(audit, "ar_replay_metric_pass")
    export_parity_pass = _audit_field_pass(audit, "export_parity_pass")

    training_allowed = bool(accepted_batches)
    ar_ready_claim_allowed = (
        bool(real_floorray_models)
        and geometry_pass
        and bbox_pass
        and ar_replay_pass
        and export_parity_pass
    )

    demo_ready = bool(
        demo_summary is not None
        and demo_summary.get("production_claim") is False
        and demo_summary.get("ar_ready_claim") is False
    )

    if ar_ready_claim_allowed and training_allowed:
        overall = OVERALL_AR_READY
    elif training_allowed:
        overall = OVERALL_TRAINING_ALLOWED_AR_BLOCKED
    elif demo_ready:
        overall = OVERALL_DEMO_READY
    else:
        overall = OVERALL_BLOCKED

    blockers: list[str] = []
    if not accepted_batches:
        if not acceptances:
            blockers.append(
                "No data/incoming/*/metadata/acceptance_status.json found; "
                "training stays blocked."
            )
        else:
            for batch in acceptances:
                if batch.derived_status != "ACCEPT_FOR_TRAINING":
                    reason = "; ".join(batch.failures) or "unknown reason"
                    blockers.append(
                        f"batch {batch.batch_id}: status={batch.derived_status} ({reason})"
                    )
    if not real_floorray_models:
        if not semantics:
            if pose_runs_present:
                blockers.append(
                    "No runs/pose/*/SEMANTICS.md found; "
                    "existing pose runs have no machine-readable freshness signal."
                )
            else:
                blockers.append(
                    "No trained pose run with SEMANTICS.md "
                    "(semantics_version=floorray_v1, trained_on_real_data=true, stale=false)."
                )
        else:
            for s in semantics:
                if not s.is_real_floorray:
                    reasons = "; ".join(s.failures) or (
                        "missing one of: semantics_version=floorray_v1, "
                        "trained_on_real_data=true, stale=false"
                    )
                    blockers.append(f"model {s.model_dir}: {reasons}")
    if stale_models:
        blockers.append(
            "stale models present: " + ", ".join(s.model_dir for s in stale_models)
        )
    if audit is None:
        blockers.append(
            "No outputs/full_pipeline_audit/REPORT.json; "
            "geometry / bbox / AR-replay / export-parity gates default to FAIL."
        )
    else:
        for key, ok in (
            ("geometry_audit_pass", geometry_pass),
            ("bbox_audit_pass", bbox_pass),
            ("ar_replay_metric_pass", ar_replay_pass),
            ("export_parity_pass", export_parity_pass),
        ):
            if not ok:
                blockers.append(f"audit gate {key} != true")

    if not training_allowed:
        if not acceptances:
            next_safe_task = (
                "Write data/incoming/android_plugin_real/metadata/"
                "acceptance_status.json using docs/DATASET_ACCEPTANCE_TEMPLATE.md "
                "once the exporter emits real WheelBBox/BBox."
            )
        elif debug_batches and not accepted_batches:
            next_safe_task = (
                "Fix the plugin/exporter so WheelBBox/BBox is real "
                "(see docs/EXPORT_PARITY_AUDIT.md), then re-issue acceptance "
                "metadata with requires_plugin_bbox=false and human preview."
            )
        elif rejected_batches and not accepted_batches:
            failing = ", ".join(b.batch_id for b in rejected_batches)
            next_safe_task = (
                f"Resolve REJECT_NEEDS_FIX on batch(es): {failing}. "
                "Human preview must pass and bbox_source must not be placeholder."
            )
        else:
            next_safe_task = (
                "Mark at least one batch as ACCEPT_FOR_TRAINING per "
                "docs/DATASET_ACCEPTANCE_TEMPLATE.md."
            )
    elif not ar_ready_claim_allowed:
        if not real_floorray_models:
            next_safe_task = (
                "Train on the accepted batch and save "
                "runs/pose/<model>/SEMANTICS.md with "
                "semantics_version=floorray_v1, trained_on_real_data=true, stale=false."
            )
        else:
            next_safe_task = (
                "Run geometry + bbox audit and AR-replay metric, then write "
                "outputs/full_pipeline_audit/REPORT.json with "
                "geometry_audit_pass / bbox_audit_pass / ar_replay_metric_pass / "
                "export_parity_pass set to true."
            )
    else:
        next_safe_task = (
            "All listed gates green. Proceed to mobile / runtime validation per "
            "docs/PRODUCTION_READINESS_PLAN.md §6."
        )

    if now is None:
        now = datetime.now(tz=timezone.utc)

    return {
        "schema_version": 1,
        "generated_at_utc": now.replace(microsecond=0).isoformat(),
        "overall_status": overall,
        "demo_ready": demo_ready,
        "training_allowed": training_allowed,
        "ar_ready_claim_allowed": ar_ready_claim_allowed,
        "production_ready": False,
        "current_blockers": blockers,
        "next_safe_task": next_safe_task,
        "inputs_seen": {
            "acceptance_files": [b.path for b in acceptances],
            "audit_report": bool(audit),
            "demo_summary": bool(demo_summary),
            "semantics_files": [s.path for s in semantics],
        },
        "batches": [b.to_dict() for b in acceptances],
        "models": [s.to_dict() for s in semantics],
        "audit": {
            "present": audit is not None,
            "geometry_audit_pass": geometry_pass,
            "bbox_audit_pass": bbox_pass,
            "ar_replay_metric_pass": ar_replay_pass,
            "export_parity_pass": export_parity_pass,
        },
        "demo_summary_seen": {
            "present": demo_summary is not None,
            "production_claim": (demo_summary or {}).get("production_claim"),
            "ar_ready_claim": (demo_summary or {}).get("ar_ready_claim"),
        },
    }


def render_markdown(verdict: dict) -> str:
    """Render the human-readable REPORT.md."""
    lines: list[str] = []
    lines.append("# Production Readiness Report")
    lines.append("")
    lines.append(f"Generated: {verdict['generated_at_utc']} UTC")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- **overall_status:** `{verdict['overall_status']}`")
    lines.append(f"- **demo_ready:** `{verdict['demo_ready']}`")
    lines.append(f"- **training_allowed:** `{verdict['training_allowed']}`")
    lines.append(f"- **ar_ready_claim_allowed:** `{verdict['ar_ready_claim_allowed']}`")
    lines.append(f"- **production_ready:** `{verdict['production_ready']}`")
    lines.append("")
    lines.append("## What is safe to show now")
    lines.append("")
    if verdict["demo_ready"]:
        lines.append(
            "- The AWE demo pack at `outputs/awe_demo/` is safe to present with the "
            "provenance badges, narration cues, and 'what to say / what not to say' "
            "section from `outputs/awe_demo/README.md`."
        )
    else:
        lines.append(
            "- No demo artefacts found. Build one with "
            "`scripts/build_awe_demo_pack.py` before presenting."
        )
    if verdict["training_allowed"]:
        lines.append(
            "- Training is allowed against at least one ACCEPT_FOR_TRAINING batch."
        )
    if verdict["ar_ready_claim_allowed"]:
        lines.append("- AR-ready claim is allowed for at least one model.")
    lines.append("")
    lines.append("## What is unsafe to claim now")
    lines.append("")
    if not verdict["ar_ready_claim_allowed"]:
        lines.append(
            "- Do **not** claim AR-ready, production-ready, or AR-3D-validated."
        )
    if not verdict["training_allowed"]:
        lines.append("- Do **not** start a training run on current data.")
    lines.append(
        "- Do **not** start MobileNetV2 / TFLite work until the gates above flip."
    )
    lines.append("- Do **not** mark any dataset `ACCEPT_FOR_TRAINING` automatically.")
    lines.append("")
    lines.append("## Current blockers")
    lines.append("")
    if not verdict["current_blockers"]:
        lines.append("- (none)")
    else:
        for b in verdict["current_blockers"]:
            lines.append(f"- {b}")
    lines.append("")
    lines.append("## Next safe task")
    lines.append("")
    lines.append(verdict["next_safe_task"])
    lines.append("")
    lines.append("## Inputs inspected")
    lines.append("")
    seen = verdict["inputs_seen"]
    lines.append(f"- Acceptance files: {len(seen['acceptance_files'])} found.")
    for p in seen["acceptance_files"]:
        lines.append(f"  - `{p}`")
    lines.append(f"- Full-pipeline audit report present: `{seen['audit_report']}`.")
    lines.append(f"- Demo summary present: `{seen['demo_summary']}`.")
    lines.append(f"- SEMANTICS.md files: {len(seen['semantics_files'])} found.")
    for p in seen["semantics_files"]:
        lines.append(f"  - `{p}`")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append(
        "- `docs/PRODUCTION_READINESS_PLAN.md` — gate definitions and roadmap."
    )
    lines.append("- `docs/DATASET_ACCEPTANCE_TEMPLATE.md` — per-batch metadata schema.")
    lines.append(
        "- `docs/MODEL_CARD_TEMPLATE.md` — per-model card and SEMANTICS.md schema."
    )
    lines.append("- `docs/AR_REPLAY_METRIC_PLAN.md` — geometry / replay audit roadmap.")
    lines.append("- `docs/AR_ML_CONTRACT.md` — frozen ML / AR JSON contract.")
    return "\n".join(lines) + "\n"


def write_reports(verdict: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "REPORT.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(verdict), encoding="utf-8")
    return json_path, md_path


def run_check(
    *,
    data_incoming_root: Path = DEFAULT_DATA_INCOMING_ROOT,
    audit_report_path: Path = DEFAULT_AUDIT_REPORT_PATH,
    demo_summary_path: Path = DEFAULT_DEMO_SUMMARY_PATH,
    runs_pose_root: Path = DEFAULT_RUNS_POSE_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    now: datetime | None = None,
    write: bool = True,
) -> dict:
    acceptances = load_acceptance_files(data_incoming_root)
    audit = load_audit_report(audit_report_path)
    demo_summary = load_demo_summary(demo_summary_path)
    semantics = load_semantics_files(runs_pose_root)
    pose_runs_present = runs_pose_root.is_dir() and any(runs_pose_root.iterdir())
    verdict = compute_verdict(
        acceptances,
        audit,
        demo_summary,
        semantics,
        pose_runs_present=pose_runs_present,
        now=now,
    )
    if write:
        write_reports(verdict, out_dir)
    return verdict


def _format_one_line(verdict: dict) -> str:
    return (
        f"{verdict['overall_status']}  "
        f"demo_ready={verdict['demo_ready']}  "
        f"training_allowed={verdict['training_allowed']}  "
        f"ar_ready_claim_allowed={verdict['ar_ready_claim_allowed']}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-incoming-root", type=Path, default=DEFAULT_DATA_INCOMING_ROOT
    )
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT_PATH)
    p.add_argument("--demo-summary", type=Path, default=DEFAULT_DEMO_SUMMARY_PATH)
    p.add_argument("--runs-pose-root", type=Path, default=DEFAULT_RUNS_POSE_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verdict = run_check(
        data_incoming_root=args.data_incoming_root,
        audit_report_path=args.audit_report,
        demo_summary_path=args.demo_summary,
        runs_pose_root=args.runs_pose_root,
        out_dir=args.out_dir,
    )
    print(_format_one_line(verdict))
    print(f"Wrote {args.out_dir / 'REPORT.json'} and {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
