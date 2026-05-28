# AR Replay Logging Harness

This harness is the runnable AR-side evidence producer for the
`ar_3d_replay_validation` production blocker.

ML returns only 2D wheel points. The AR app owns camera pose capture,
floor raycasts, RANSAC/plane recovery, and final disc-bottom 3D
reconstruction. This harness writes those AR-side observations to the
JSONL format consumed by `src/validate_ar_replay.py`.

## Files To Copy

Copy or adapt:

```text
ar_replay_harness/ArReplayLogger.kt
```

Recommended Android target path:

```text
app/src/main/java/com/vsbl/wheels/ar/ArReplayLogger.kt
```

Change only the package line if the Android app uses a different package.

## Integration Shape

Create one logger per capture session:

```kotlin
val logger = ArReplayLogger(
    context = context,
    sessionId = "s_2026_05_27_001",
    captureDevice = "${Build.MANUFACTURER} ${Build.MODEL}",
    captureAppVersion = BuildConfig.VERSION_NAME,
    captureDateUtc = "2026-05-27"
)
```

For every wheel observation, pass the exact ML screen-space points and
the AR-side raycast/RANSAC results:

```kotlin
logger.appendObservation(
    frameId = frameId,
    captureIndex = frameIndex,
    wheelIndex = wheelIndex,
    cameraTransform = ArReplayLogger.CameraTransform(
        rotation = listOf(
            listOf(r00, r01, r02),
            listOf(r10, r11, r12),
            listOf(r20, r21, r22)
        ),
        translation = ArReplayLogger.Vec3(tx, ty, tz)
    ),
    points = ArReplayLogger.ScreenPoints(
        a = ArReplayLogger.Vec2(aX, aY),
        b = ArReplayLogger.Vec2(bX, bY),
        cDiscBottom = ArReplayLogger.Vec2(cX, cY)
    ),
    floorHits = ArReplayLogger.FloorRaycastHits(
        a = ArReplayLogger.Vec3(aWorldX, aWorldY, aWorldZ),
        b = ArReplayLogger.Vec3(bWorldX, bWorldY, bWorldZ)
    ),
    ransac = ArReplayLogger.RansacResult(
        inlier = true,
        residual = residual,
        recoveredPlane = ArReplayLogger.RecoveredPlane(
            normal = ArReplayLogger.Vec3(nx, ny, nz),
            point = ArReplayLogger.Vec3(px, py, pz),
            support = support
        ),
        cPlaneHit = ArReplayLogger.Vec3(cWorldX, cWorldY, cWorldZ),
        cHeightValue = cHeight,
        finalDiscBottomPosition = ArReplayLogger.Vec3(finalX, finalY, finalZ)
    )
)
```

If ML returns multiple wheels for the same `frameId` / `captureIndex`,
write one row per wheel and pass a unique `wheelIndex` or
`wheelTrackId` for each row. The logger fails fast on decreasing
`captureIndex`, a repeated `captureIndex` with a different `frameId`,
missing wheel identity on duplicate frame rows, and duplicate wheel
identity on one frame.

The file is written to:

```text
<app external files dir>/ar_replay.jsonl
```

Pull or copy it into this ML repo:

```text
data/incoming/ar_3d_replay/ar_replay.jsonl
```

Then validate:

```bash
./.venv/bin/python src/validate_ar_replay.py \
  --jsonl data/incoming/ar_3d_replay/ar_replay.jsonl \
  --out outputs/production_audit/ar_3d_replay_eval.json
```

## Acceptance

The production validator requires at least 30 valid observations, real
`schema_version=1`, production `source_type`, non-placeholder
`capture_device`, camera pose evidence (`cameraTransform` or
`cameraPoseRef`), non-placeholder
`capture_app_version`, a real `capture_date_utc` date in `YYYY-MM-DD`
format that is not in the future, complete A/B floor raycast hits,
RANSAC labels, non-negative residuals, recovered plane evidence with a
unit normal and positive support, `cPlaneHit`, non-negative
`cHeightValue`, and at least one final disc-bottom 3D position. The
logger fails fast when these production fields are missing or invalid.

See `docs/AR_MOCK_LOG_CONTRACT.md` for the full schema and thresholds.
