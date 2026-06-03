# External Evidence Handoff Bundle

Build the Android/AR handoff zip:

```bash
./.venv/bin/python scripts/build_external_evidence_handoff_bundle.py
```

Outputs:

```text
outputs/production_audit/external_evidence_handoff_bundle.zip
outputs/production_audit/external_evidence_handoff_bundle_manifest.json
outputs/production_audit/external_evidence_handoff_bundle_verification.json
```

The bundle contains the exact Android TFLite artifact, the iOS CoreML
`.mlmodel` artifact, the iOS CoreML README/Swift smoke harness, the
Android LiteRT instrumentation-test harness, the
AR holdout annotation writer, the AR replay logging harness, evidence
templates, production contracts, validators, and the one-command evidence
intake runner. It is for Android/iOS/AR integration and data collection
handoff; it does not certify production by itself.

For an iOS-only package with a compiled `.mlmodelc`, run:

```bash
./.venv/bin/python scripts/build_ios_coreml_handoff.py
```

It writes:

```text
outputs/production_audit/ios_coreml_handoff.zip
outputs/production_audit/ios_coreml_handoff_manifest.json
```

When Android/AR returns a zip or folder with real evidence, import it
with:

```bash
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --overwrite
```

Then run the one-command intake:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --evidence-drop path/to/evidence_drop.zip --evidence-drop-overwrite
```

To import, validate, and immediately run the final production audit suite
with pytest after a green intake:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --evidence-drop path/to/evidence_drop.zip --evidence-drop-overwrite --finalize
```

Finalization rereads the canonical `data/incoming/...` evidence files
and `outputs/production_audit/...` reports. Do not use custom paths for
the final certification run.

The intake keeps two levels of custody:

- `external_evidence_drop_import.json` records copied files and
  `evidence_manifest_sha256`.
- Validator outputs record `source_sha256` for Android LiteRT and AR
  replay, and `source_manifest_sha256` for the AR holdout directory.

The production evidence audit recomputes these values and rejects stale
or mismatched validation reports.

An empty return-zip shape is generated at:

```text
outputs/production_audit/external_evidence_return_template.zip
outputs/production_audit/external_evidence_return_template_manifest.json
```

Verify the zip against the manifest:

```bash
./.venv/bin/python src/verify_external_evidence_handoff_bundle.py
```

Verification also rehashes the current workspace copies of every
manifest artifact. A handoff bundle is rejected if the zip still matches
its manifest but any source contract, harness, template, validator,
TFLite artifact, or CoreML artifact changed after the bundle was built.
