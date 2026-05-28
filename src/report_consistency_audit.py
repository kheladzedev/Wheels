"""Check final audit/report consistency after release generation.

This is a post-release meta-audit. It does not decide production
readiness; it checks that the generated evidence is internally
consistent and avoids stale or cyclic release-report claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from src.release_integrity import PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS
    from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
    from scripts.create_external_evidence_return_template import build_template_files
except ImportError:  # pragma: no cover - used when executed as src/report_consistency_audit.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.release_integrity import PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS
    from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
    from scripts.create_external_evidence_return_template import build_template_files


DEFAULT_JSON_OUT = Path("outputs/production_audit/report_consistency_audit.json")
DEFAULT_MD_OUT = Path("docs/REPORT_CONSISTENCY_AUDIT.md")
EXPECTED_ANDROID_TFLITE = Path("outputs/production_audit/tflite_export/best_float32.tflite")
EXPECTED_RETURN_TEMPLATE_ZIP = Path("outputs/production_audit/external_evidence_return_template.zip")
EXPECTED_RETURN_TEMPLATE_MANIFEST = Path(
    "outputs/production_audit/external_evidence_return_template_manifest.json"
)
EXPECTED_ANDROID_ARTIFACT_FORMAT = "tflite_float32"
EXPECTED_ANDROID_ARTIFACT_ENTRY = "EXPECTED_ANDROID_ARTIFACT.json"
SELF_REFERENTIAL_RELEASE_PATHS = {
    "outputs/production_audit/release_integrity.json",
    "docs/RELEASE_PACKAGE.md",
}
PRE_FINAL_RELEASE_COUNT_DOCS = [
    Path("docs/REQUIREMENTS_TRACEABILITY.md"),
    Path("docs/EXECUTIVE_REPORT_RU.md"),
    Path("docs/MODEL_CARD.md"),
]
STALE_RELEASE_PATTERNS = [
    "Release artifacts:",
    "Release size:",
    "Artifacts: 72",
    "Total size: 125.938",
    '"release_artifacts": 72',
]
EXPECTED_FINALIZATION_COMMAND = [
    "./.venv/bin/python",
    "src/production_audit_suite.py",
    "--with-pytest",
]
EXPECTED_POST_FINALIZATION_REFRESH_COMMANDS = [
    ["./.venv/bin/python", "scripts/write_production_audit_report.py"],
    ["./.venv/bin/python", "scripts/write_handoff_report.py"],
    ["./.venv/bin/python", "src/release_integrity.py"],
    ["./.venv/bin/python", "src/report_consistency_audit.py"],
]


@dataclass
class ConsistencyCheck:
    name: str
    ok: bool
    detail: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_zip_json(path: Path, entry: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open(entry) as f:
                payload = json.loads(f.read().decode("utf-8"))
    except (OSError, KeyError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_release_manifest_no_self_reference(release: dict[str, Any]) -> ConsistencyCheck:
    paths = {
        str(artifact.get("path", ""))
        for artifact in release.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    bad = sorted(paths & SELF_REFERENTIAL_RELEASE_PATHS)
    return ConsistencyCheck(
        "release_manifest_no_self_reference",
        not bad and bool(release.get("ok")),
        f"self_references={bad}, release_ok={release.get('ok', 'n/a')}",
    )


def check_release_manifest_hashes_current(release: dict[str, Any]) -> ConsistencyCheck:
    mismatches: list[str] = []
    missing: list[str] = []
    for artifact in release.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        path_value = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(path_value, str) or not expected_sha:
            continue
        path = Path(path_value)
        actual_sha = sha256_file(path)
        if actual_sha is None:
            missing.append(path_value)
        elif actual_sha != expected_sha:
            mismatches.append(path_value)
    ok = not missing and not mismatches and bool(release.get("ok"))
    return ConsistencyCheck(
        "release_manifest_hashes_current",
        ok,
        f"missing={missing}, mismatches={mismatches}",
    )


def check_suite_release_count_matches_manifest(
    suite: dict[str, Any], release: dict[str, Any]
) -> ConsistencyCheck:
    suite_count = suite.get("release_artifacts")
    release_count = release.get("artifact_count")
    return ConsistencyCheck(
        "suite_release_count_matches_manifest",
        suite_count == release_count,
        f"suite_release_artifacts={suite_count}, release_artifact_count={release_count}",
    )


def check_objective_matches_suite(
    objective: dict[str, Any], suite: dict[str, Any]
) -> ConsistencyCheck:
    fields = ("integration_ready", "production_ready")
    mismatches = [
        field
        for field in fields
        if bool(objective.get(field)) != bool(suite.get(field))
    ]
    return ConsistencyCheck(
        "objective_status_matches_suite",
        not mismatches,
        f"mismatches={mismatches}",
    )


def check_objective_completion_is_consistent(objective: dict[str, Any]) -> ConsistencyCheck:
    complete = bool(objective.get("objective_complete"))
    production_ready = bool(objective.get("production_ready"))
    failed = objective.get("failed_requirements", [])
    ok = (not complete) or (production_ready and failed == [])
    return ConsistencyCheck(
        "objective_completion_consistent",
        ok,
        f"objective_complete={complete}, production_ready={production_ready}, failed_requirements={failed}",
    )


def check_gate_alignment(
    suite: dict[str, Any],
    integration_gate: dict[str, Any],
    production_gate: dict[str, Any],
) -> ConsistencyCheck:
    integration_ok = bool(integration_gate.get("ok"))
    production_ok = bool(production_gate.get("ok"))
    mismatches: list[str] = []
    if bool(suite.get("integration_ready")) != integration_ok:
        mismatches.append("integration_ready")
    if bool(suite.get("production_ready")) and not production_ok:
        mismatches.append("production_ready_without_gate")
    return ConsistencyCheck(
        "suite_status_matches_gates",
        not mismatches,
        (
            f"mismatches={mismatches}, integration_gate_ok={integration_ok}, "
            f"production_gate_ok={production_ok}"
        ),
    )


def check_no_prefinal_release_counts() -> ConsistencyCheck:
    hits: list[str] = []
    for path in PRE_FINAL_RELEASE_COUNT_DOCS:
        content = text(path)
        for pattern in STALE_RELEASE_PATTERNS:
            if pattern in content:
                hits.append(f"{path}:{pattern}")
    return ConsistencyCheck(
        "prefinal_reports_do_not_embed_release_counts",
        not hits,
        f"hits={hits}",
    )


def check_required_final_reports_present() -> ConsistencyCheck:
    required = [
        Path("docs/PRODUCTION_READINESS_AUDIT.md"),
        Path("docs/HANDOFF_TODAY.md"),
        Path("docs/OBJECTIVE_COMPLETION_AUDIT.md"),
        Path("docs/MODEL_SELECTION_AUDIT.md"),
        Path("docs/SPEC_COMPLIANCE_AUDIT.md"),
        Path("docs/RELEASE_PACKAGE.md"),
        Path("outputs/production_audit/model_selection_audit.json"),
        Path("outputs/production_audit/spec_compliance_audit.json"),
        Path("outputs/production_audit/model_package_manifest.json"),
        Path("outputs/production_audit/release_integrity.json"),
        Path("outputs/production_audit/audit_suite_status.json"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return ConsistencyCheck(
        "required_final_reports_present",
        not missing,
        f"missing={missing}",
    )


def check_model_package_manifest_artifacts(model_package: dict[str, Any]) -> ConsistencyCheck:
    artifacts = model_package.get("package_artifacts", [])
    missing: list[str] = []
    mismatches: list[str] = []
    if not isinstance(artifacts, list) or not artifacts:
        return ConsistencyCheck(
            "model_package_manifest_artifacts_current",
            False,
            "package_artifacts=missing_or_empty",
        )
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path_value = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(path_value, str) or not expected_sha:
            missing.append(str(path_value or "missing_path"))
            continue
        actual_sha = sha256_file(Path(path_value))
        if actual_sha is None:
            missing.append(path_value)
        elif actual_sha != expected_sha:
            mismatches.append(path_value)
    canonical = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    digest_ok = digest == model_package.get("package_digest_sha256")
    missing_declared = model_package.get("missing_artifacts")
    missing_declared_ok = missing_declared == []
    ok = not missing and not mismatches and digest_ok and missing_declared_ok
    return ConsistencyCheck(
        "model_package_manifest_artifacts_current",
        ok,
        (
            f"missing={missing}, mismatches={mismatches}, "
            f"digest_ok={digest_ok}, missing_declared={missing_declared}"
        ),
    )


def check_return_template_expected_android_artifact(
    manifest: dict[str, Any],
) -> ConsistencyCheck:
    expected_path = str(EXPECTED_ANDROID_TFLITE).replace("\\", "/")
    actual_tflite_sha = sha256_file(EXPECTED_ANDROID_TFLITE)
    failures: list[str] = []

    if actual_tflite_sha is None:
        failures.append("missing_expected_tflite")

    manifest_artifact = manifest.get("expected_android_artifact")
    if not isinstance(manifest_artifact, dict):
        failures.append("manifest_expected_android_artifact_missing")
        manifest_artifact = {}

    manifest_path = manifest_artifact.get("path")
    manifest_sha = manifest_artifact.get("sha256")
    manifest_format = manifest_artifact.get("format")
    if manifest_path != expected_path:
        failures.append("manifest_artifact_path_mismatch")
    if manifest_format != EXPECTED_ANDROID_ARTIFACT_FORMAT:
        failures.append("manifest_artifact_format_mismatch")
    if actual_tflite_sha is not None and manifest_sha != actual_tflite_sha:
        failures.append("manifest_artifact_sha_mismatch")

    template_value = manifest.get("template")
    template_path = Path(template_value) if isinstance(template_value, str) else EXPECTED_RETURN_TEMPLATE_ZIP
    if str(template_path).replace("\\", "/") != str(EXPECTED_RETURN_TEMPLATE_ZIP):
        failures.append("template_path_mismatch")

    actual_zip_sha = sha256_file(template_path)
    actual_zip_size = template_path.stat().st_size if template_path.is_file() else None
    if actual_zip_sha is None:
        failures.append("missing_template_zip")
    elif manifest.get("zip_sha256") != actual_zip_sha:
        failures.append("template_zip_sha_mismatch")
    if actual_zip_size is None or manifest.get("zip_size_bytes") != actual_zip_size:
        failures.append("template_zip_size_mismatch")

    manifest_artifacts = {
        str(artifact.get("path")): artifact
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("path")
    }
    if manifest.get("artifact_count") != len(manifest_artifacts):
        failures.append("manifest_artifact_count_mismatch")
    expected_entries = sorted(build_template_files(EXPECTED_ANDROID_TFLITE))
    try:
        with zipfile.ZipFile(template_path) as zf:
            zip_entries = sorted(info.filename for info in zf.infolist() if not info.is_dir())
            if zip_entries != expected_entries:
                failures.append("template_zip_entries_mismatch")
            if sorted(manifest_artifacts) != zip_entries:
                failures.append("template_manifest_entries_mismatch")
            for entry in zip_entries:
                payload = zf.read(entry)
                artifact = manifest_artifacts.get(entry, {})
                if artifact.get("sha256") != sha256_bytes(payload):
                    failures.append(f"template_manifest_entry_sha_mismatch:{entry}")
                if artifact.get("size_bytes") != len(payload):
                    failures.append(f"template_manifest_entry_size_mismatch:{entry}")
    except (OSError, zipfile.BadZipFile):
        failures.append("template_zip_unreadable")

    embedded = read_zip_json(template_path, EXPECTED_ANDROID_ARTIFACT_ENTRY)
    embedded_artifact = embedded.get("expected_android_artifact")
    if not isinstance(embedded_artifact, dict):
        failures.append("embedded_expected_android_artifact_missing")
        embedded_artifact = {}
    if embedded_artifact != manifest_artifact:
        failures.append("embedded_artifact_manifest_mismatch")
    if actual_tflite_sha is not None and embedded_artifact.get("sha256") != actual_tflite_sha:
        failures.append("embedded_artifact_sha_mismatch")

    return ConsistencyCheck(
        "external_evidence_return_template_artifact_current",
        not failures,
        (
            f"failures={failures}, artifact_path={manifest_path}, "
            f"manifest_sha={manifest_sha}, actual_sha={actual_tflite_sha}, "
            f"zip_sha_match={actual_zip_sha is not None and manifest.get('zip_sha256') == actual_zip_sha}, "
            f"zip_size_match={actual_zip_size is not None and manifest.get('zip_size_bytes') == actual_zip_size}"
        ),
    )


def check_intake_finalization_contract(
    intake: dict[str, Any],
    preflight: dict[str, Any],
    production_evidence: dict[str, Any],
) -> ConsistencyCheck:
    doc = text(Path("docs/PRODUCTION_EVIDENCE_INTAKE.md"))
    command_in_doc = "./.venv/bin/python src/production_audit_suite.py --with-pytest" in doc
    finalize_in_doc = "--finalize" in doc
    finalization_required_in_doc = "finalization_required=true" in doc
    production_ready = production_evidence.get("production_evidence_ready") is True
    status = intake if production_ready else preflight
    status_source = "intake" if production_ready else "preflight"
    finalization = status.get("finalization")
    finalization_ok = isinstance(finalization, dict) and finalization.get("ok") is True
    pending_finalization_ok = (
        status.get("finalization_required") is True
        and status.get("finalization_command") == EXPECTED_FINALIZATION_COMMAND
    )
    completed_finalization_ok = (
        status.get("finalization_required") is False
        and status.get("finalization_command") == EXPECTED_FINALIZATION_COMMAND
        and finalization_ok
    )
    ok = (
        (pending_finalization_ok or completed_finalization_ok)
        and command_in_doc
        and finalize_in_doc
        and finalization_required_in_doc
    )
    return ConsistencyCheck(
        "intake_finalization_contract",
        ok,
        (
            f"status_source={status_source}, "
            f"production_evidence_ready={production_ready}, "
            f"status_finalization_required={status.get('finalization_required', 'n/a')}, "
            f"status_finalization_command={status.get('finalization_command', 'n/a')}, "
            f"status_finalization_ok={finalization_ok}, "
            f"doc_command={command_in_doc}, doc_finalize={finalize_in_doc}, "
            f"doc_finalization_required={finalization_required_in_doc}"
        ),
    )


def _command_suffix(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    values = [str(item) for item in command]
    return values[1:] if values and values[0].endswith("python") else values


def check_intake_post_finalization_refresh_contract(
    intake: dict[str, Any],
    production_evidence: dict[str, Any],
) -> ConsistencyCheck:
    production_ready = production_evidence.get("production_evidence_ready") is True
    finalization = intake.get("finalization")
    finalization_ok = isinstance(finalization, dict) and finalization.get("ok") is True
    required = production_ready and finalization_ok
    refresh = intake.get("post_finalization_refresh", [])
    failures: list[str] = []
    if required:
        if not isinstance(refresh, list):
            failures.append("post_finalization_refresh_missing")
            refresh = []
        expected_suffixes = [_command_suffix(command) for command in EXPECTED_POST_FINALIZATION_REFRESH_COMMANDS]
        actual_suffixes = [_command_suffix(item.get("command")) for item in refresh if isinstance(item, dict)]
        if actual_suffixes != expected_suffixes:
            failures.append("post_finalization_refresh_command_mismatch")
        for index, item in enumerate(refresh):
            if not isinstance(item, dict):
                failures.append(f"post_finalization_refresh_item_not_object:{index}")
                continue
            if item.get("ok") is not True or item.get("returncode") != 0:
                failures.append(f"post_finalization_refresh_not_ok:{index}:{item.get('name', 'missing')}")
    return ConsistencyCheck(
        "intake_post_finalization_refresh_contract",
        not failures,
        (
            f"required={required}, production_evidence_ready={production_ready}, "
            f"finalization_ok={finalization_ok}, refresh_count={len(refresh) if isinstance(refresh, list) else 'n/a'}, "
            f"failures={failures}"
        ),
    )


def check_handoff_bundle_verification(verification: dict[str, Any]) -> ConsistencyCheck:
    required_count = len(DEFAULT_BUNDLE_ARTIFACTS)
    entry_count = verification.get("entry_count")
    expected_entry_count = verification.get("expected_entry_count")
    verified_required_count = verification.get("required_artifact_count")
    missing_required = verification.get("missing_required_artifacts", [])
    failures = verification.get("failures", [])
    bundle_sha = verification.get("bundle_sha256")
    expected_bundle_sha = verification.get("expected_bundle_sha256")
    bundle_size = verification.get("bundle_size_bytes")
    expected_bundle_size = verification.get("expected_bundle_size_bytes")
    current_artifacts = verification.get("current_artifacts", [])
    current_artifacts_ok = (
        isinstance(current_artifacts, list)
        and len(current_artifacts) == required_count
        and all(isinstance(item, dict) and item.get("ok") is True for item in current_artifacts)
    )
    ok = (
        verification.get("ok") is True
        and verified_required_count == required_count
        and entry_count == expected_entry_count == required_count
        and current_artifacts_ok
        and isinstance(bundle_sha, str)
        and bool(bundle_sha)
        and bundle_sha == expected_bundle_sha
        and isinstance(bundle_size, int)
        and bundle_size > 0
        and bundle_size == expected_bundle_size
        and missing_required == []
        and failures == []
    )
    return ConsistencyCheck(
        "handoff_bundle_verification_strict",
        ok,
        (
            f"verification_ok={verification.get('ok', 'n/a')}, "
            f"required_artifact_count={verified_required_count}/{required_count}, "
            f"entry_count={entry_count}, expected_entry_count={expected_entry_count}, "
            f"current_artifacts_ok={current_artifacts_ok}, "
            f"bundle_sha_match={bundle_sha == expected_bundle_sha}, "
            f"bundle_size_match={bundle_size == expected_bundle_size}, "
            f"missing_required={missing_required}, failures={failures}"
        ),
    )


def check_production_evidence_release_artifacts(
    production_evidence: dict[str, Any],
    release: dict[str, Any],
) -> ConsistencyCheck:
    release_paths = {
        str(artifact.get("path", ""))
        for artifact in release.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    required = bool(production_evidence.get("production_evidence_ready"))
    missing = sorted(
        artifact
        for artifact in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS
        if artifact not in release_paths
    )
    ok = (not required) or not missing
    return ConsistencyCheck(
        "production_evidence_release_artifacts",
        ok,
        (
            f"production_evidence_ready={required}, "
            f"required_artifacts={len(PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS)}, "
            f"missing={missing}"
        ),
    )

def build_audit() -> dict[str, Any]:
    suite = read_json(Path("outputs/production_audit/audit_suite_status.json"))
    release = read_json(Path("outputs/production_audit/release_integrity.json"))
    objective = read_json(Path("outputs/production_audit/objective_completion_audit.json"))
    integration_gate = read_json(Path("outputs/production_audit/integration_gate.json"))
    production_gate = read_json(Path("outputs/production_audit/production_gate.json"))
    production_evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    intake = read_json(Path("outputs/production_audit/production_evidence_intake_status.json"))
    preflight = read_json(Path("outputs/production_audit/production_evidence_intake_preflight_status.json"))
    model_package = read_json(Path("outputs/production_audit/model_package_manifest.json"))
    handoff_verification = read_json(
        Path("outputs/production_audit/external_evidence_handoff_bundle_verification.json")
    )
    return_template_manifest = read_json(EXPECTED_RETURN_TEMPLATE_MANIFEST)

    checks = [
        check_release_manifest_no_self_reference(release),
        check_release_manifest_hashes_current(release),
        check_suite_release_count_matches_manifest(suite, release),
        check_objective_matches_suite(objective, suite),
        check_objective_completion_is_consistent(objective),
        check_gate_alignment(suite, integration_gate, production_gate),
        check_no_prefinal_release_counts(),
        check_required_final_reports_present(),
        check_model_package_manifest_artifacts(model_package),
        check_return_template_expected_android_artifact(return_template_manifest),
        check_intake_finalization_contract(intake, preflight, production_evidence),
        check_intake_post_finalization_refresh_contract(intake, production_evidence),
        check_handoff_bundle_verification(handoff_verification),
        check_production_evidence_release_artifacts(production_evidence, release),
    ]
    failures = [check.name for check in checks if not check.ok]
    return {
        "schema_version": 1,
        "ok": not failures,
        "failures": failures,
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Report Consistency Audit",
        "",
        f"- OK: {audit['ok']}",
        f"- Failures: {', '.join(audit['failures']) if audit['failures'] else 'none'}",
        "",
        "| Check | OK | Detail |",
        "|---|---:|---|",
    ]
    for check in audit["checks"]:
        lines.append(f"| {check['name']} | {check['ok']} | {check['detail']} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"ok={audit['ok']} failures={audit['failures']}")
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
