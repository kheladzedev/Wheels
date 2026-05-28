package com.vsbl.wheels.ar

import android.content.Context
import java.io.File
import java.time.LocalDate
import java.time.ZoneOffset
import java.time.format.DateTimeParseException
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

private const val UNIT_NORMAL_TOLERANCE = 0.05

class ArReplayLogger(
    context: Context,
    private val sessionId: String,
    private val captureDevice: String,
    private val captureAppVersion: String,
    private val captureDateUtc: String,
    private val sourceType: String = SOURCE_TYPE_ANDROID_AR_DEVICE_REPLAY,
    fileName: String = DEFAULT_FILE_NAME
) {
    private val outFile: File = File(
        requireNotNull(context.getExternalFilesDir(null)) {
            "External files directory is unavailable"
        },
        fileName
    )

    fun file(): File = outFile
    private var lastCaptureIndex: Int? = null
    private var lastFrameId: String? = null
    private val wheelIdentitiesByFrame = mutableMapOf<String, MutableSet<String?>>()

    fun reset() {
        outFile.parentFile?.mkdirs()
        outFile.writeText("")
        lastCaptureIndex = null
        lastFrameId = null
        wheelIdentitiesByFrame.clear()
    }

    fun appendObservation(
        frameId: String,
        captureIndex: Int,
        cameraTransform: CameraTransform?,
        cameraPoseRef: String? = null,
        points: ScreenPoints,
        floorHits: FloorRaycastHits,
        ransac: RansacResult? = null,
        wheelIndex: Int? = null,
        wheelTrackId: String? = null
    ) {
        require(sessionId.isNotBlank()) { "sessionId must not be blank" }
        require(frameId.isNotBlank()) { "frameId must not be blank" }
        require(captureDevice.isNotBlank()) { "captureDevice must not be blank" }
        require(captureAppVersion.isNotBlank()) { "captureAppVersion must not be blank" }
        require(isValidUtcDate(captureDateUtc)) {
            "captureDateUtc must be a real UTC date in YYYY-MM-DD format and must not be in the future"
        }
        require(captureIndex >= 0) { "captureIndex must be non-negative" }
        require(wheelIndex == null || wheelIndex >= 0) {
            "wheelIndex must be non-negative when present"
        }
        require(wheelTrackId == null || wheelTrackId.isNotBlank()) {
            "wheelTrackId must be non-blank when present"
        }
        require(wheelIndex == null || wheelTrackId == null) {
            "use only one of wheelIndex or wheelTrackId"
        }
        require((cameraTransform == null) xor (cameraPoseRef?.isNotBlank() == true)) {
            "exactly one of cameraTransform or non-blank cameraPoseRef is required"
        }
        require(floorHits.a != null && floorHits.b != null) {
            "floorHits.a and floorHits.b are required for production replay evidence"
        }
        require(ransac != null) {
            "ransac result is required for production replay evidence"
        }
        validateCaptureOrder(frameId, captureIndex, wheelIndex, wheelTrackId)

        val observation = JSONObject()
            .put("schema_version", 1)
            .put("source_type", sourceType)
            .put("capture_device", captureDevice)
            .put("capture_app_version", captureAppVersion)
            .put("capture_date_utc", captureDateUtc)
            .put("session_id", sessionId)
            .put("frame_id", frameId)
            .put("capture_index", captureIndex)
            .put("camera_transform", cameraTransform?.toJson() ?: JSONObject.NULL)
            .put("camera_pose_ref", cameraPoseRef ?: JSONObject.NULL)
            .put("screen_points", points.toJson())
            .put("floor_raycast_hits", floorHits.toJson())

        wheelIndex?.let { observation.put("wheel_index", it) }
        wheelTrackId?.let { observation.put("wheel_track_id", it) }

        observation
            .put("inlier", ransac.inlier)
            .put("residual", ransac.residual)
            .put("recovered_plane", ransac.recoveredPlane.toJson())
            .put("c_plane_hit", ransac.cPlaneHit.toJsonArray())
            .put("c_height_value", ransac.cHeightValue)
            .put(
                "final_disc_bottom_position",
                ransac.finalDiscBottomPosition?.toJsonArray() ?: JSONObject.NULL
            )

        outFile.parentFile?.mkdirs()
        outFile.appendText(observation.toString() + "\n")
    }

    private fun validateCaptureOrder(
        frameId: String,
        captureIndex: Int,
        wheelIndex: Int?,
        wheelTrackId: String?
    ) {
        val previousIndex = lastCaptureIndex
        if (previousIndex != null) {
            require(captureIndex >= previousIndex) {
                "captureIndex must be non-decreasing within one session"
            }
            require(captureIndex != previousIndex || frameId == lastFrameId) {
                "repeated captureIndex must keep the same frameId"
            }
        }

        val frameKey = "$captureIndex::$frameId"
        val identities = wheelIdentitiesByFrame.getOrPut(frameKey) { mutableSetOf() }
        val identity = when {
            wheelIndex != null -> "wheel_index:$wheelIndex"
            wheelTrackId != null -> "wheel_track_id:$wheelTrackId"
            else -> null
        }
        if (identities.isNotEmpty()) {
            require(identity != null && !identities.contains(null)) {
                "repeated frame/captureIndex rows require wheelIndex or wheelTrackId"
            }
            require(!identities.contains(identity)) {
                "repeated frame/captureIndex rows require unique wheel identity"
            }
        }
        identities.add(identity)
        lastCaptureIndex = captureIndex
        lastFrameId = frameId
    }

    data class Vec2(val x: Double, val y: Double) {
        init {
            require(x.isFinite() && y.isFinite()) { "Vec2 values must be finite" }
        }

        fun toJsonArray(): JSONArray = JSONArray(listOf(x, y))
    }

    data class Vec3(val x: Double, val y: Double, val z: Double) {
        init {
            require(x.isFinite() && y.isFinite() && z.isFinite()) {
                "Vec3 values must be finite"
            }
        }

        fun toJsonArray(): JSONArray = JSONArray(listOf(x, y, z))
    }

    data class ScreenPoints(
        val a: Vec2,
        val b: Vec2,
        val cDiscBottom: Vec2
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("a", a.toJsonArray())
            .put("b", b.toJsonArray())
            .put("c_disc_bottom", cDiscBottom.toJsonArray())
    }

    data class FloorRaycastHits(
        val a: Vec3?,
        val b: Vec3?
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("a", a?.toJsonArray() ?: JSONObject.NULL)
            .put("b", b?.toJsonArray() ?: JSONObject.NULL)
    }

    data class CameraTransform(
        val rotation: List<List<Double>>,
        val translation: Vec3
    ) {
        init {
            require(rotation.size == 3 && rotation.all { it.size == 3 }) {
                "rotation must be a 3x3 matrix"
            }
            require(rotation.flatten().all { it.isFinite() }) {
                "rotation values must be finite"
            }
        }

        fun toJson(): JSONObject = JSONObject()
            .put("R", JSONArray(rotation.map { JSONArray(it) }))
            .put("t", translation.toJsonArray())
    }

    data class RecoveredPlane(
        val normal: Vec3,
        val point: Vec3,
        val support: Int
    ) {
        init {
            require(support > 0) { "support must be positive for production replay evidence" }
            val norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
            require(kotlin.math.abs(norm - 1.0) <= UNIT_NORMAL_TOLERANCE) {
                "recoveredPlane.normal must be a unit vector"
            }
        }

        fun toJson(): JSONObject = JSONObject()
            .put("normal", normal.toJsonArray())
            .put("point", point.toJsonArray())
            .put("support", support)
    }

    data class RansacResult(
        val inlier: Boolean,
        val residual: Double,
        val recoveredPlane: RecoveredPlane,
        val cPlaneHit: Vec3,
        val cHeightValue: Double,
        val finalDiscBottomPosition: Vec3?
    ) {
        init {
            require(residual.isFinite() && residual >= 0.0) {
                "residual must be finite and non-negative"
            }
            require(cHeightValue.isFinite()) {
                "cHeightValue must be finite for production replay evidence"
            }
            require(cHeightValue >= 0.0) {
                "cHeightValue must be non-negative for production replay evidence"
            }
        }
    }

    companion object {
        const val SOURCE_TYPE_ANDROID_AR_DEVICE_REPLAY = "android_ar_device_replay"
        const val DEFAULT_FILE_NAME = "ar_replay.jsonl"

        private fun isValidUtcDate(value: String): Boolean {
            if (!Regex("^\\d{4}-\\d{2}-\\d{2}$").matches(value)) {
                return false
            }
            return try {
                val parsed = LocalDate.parse(value)
                !parsed.isAfter(LocalDate.now(ZoneOffset.UTC))
            } catch (_: DateTimeParseException) {
                false
            }
        }
    }
}
