package com.vsbl.wheels.ar

import android.content.Context
import java.io.File
import java.time.LocalDate
import java.time.ZoneOffset
import java.time.format.DateTimeParseException
import org.json.JSONArray
import org.json.JSONObject

class ArHoldoutAnnotationWriter(
    context: Context,
    rootName: String = DEFAULT_ROOT_NAME
) {
    private val root: File = File(
        requireNotNull(context.getExternalFilesDir(null)) {
            "External files directory is unavailable"
        },
        rootName
    )
    private val imagesDir = File(root, "images")
    private val annotationsDir = File(root, "annotations")
    private val metadataDir = File(root, "metadata")

    fun root(): File = root

    fun reset() {
        if (root.exists()) {
            root.deleteRecursively()
        }
        imagesDir.mkdirs()
        annotationsDir.mkdirs()
        metadataDir.mkdirs()
    }

    fun writeProvenance(
        captureDevice: String,
        captureAppVersion: String,
        captureDateUtc: String,
        annotator: String,
        reviewer: String,
        notes: String? = null
    ) {
        require(captureDevice.isNotBlank()) { "captureDevice must not be blank" }
        require(captureAppVersion.isNotBlank()) { "captureAppVersion must not be blank" }
        require(captureDateUtc.isNotBlank()) { "captureDateUtc must not be blank" }
        require(annotator.isNotBlank()) { "annotator must not be blank" }
        require(reviewer.isNotBlank()) { "reviewer must not be blank" }
        require(annotator != reviewer) {
            "annotator and reviewer must be different people/accounts"
        }
        require(isValidUtcDate(captureDateUtc)) {
            "captureDateUtc must be a real UTC date in YYYY-MM-DD format and must not be in the future"
        }

        metadataDir.mkdirs()
        val payload = JSONObject()
            .put("schema_version", 1)
            .put("source_type", SOURCE_TYPE_ANDROID_AR_DEVICE_HUMAN_LABELLED)
            .put("label_type", LABEL_TYPE_HUMAN_REVIEWED)
            .put("capture_device", captureDevice)
            .put("review_status", REVIEW_STATUS_ACCEPTED)
            .put("capture_app_version", captureAppVersion)
            .put("capture_date_utc", captureDateUtc)
            .put("annotator", annotator)
            .put("reviewer", reviewer)
            .put("notes", notes ?: "")

        File(metadataDir, "provenance.json").writeText(payload.toString(2))
    }

    fun writeFrame(
        frameId: String,
        imageFileName: String,
        imageBytes: ByteArray,
        wheels: List<Wheel>
    ) {
        requireValidFrameId(frameId)
        require(!imageFileName.contains("/") && !imageFileName.contains("\\")) {
            "imageFileName must be a filename, not a path"
        }
        require(imageFileName.startsWith(frameId)) {
            "imageFileName should share the frameId stem"
        }
        require(imageFileName.substringAfterLast('.', "").lowercase() in SUPPORTED_IMAGE_EXTENSIONS) {
            "imageFileName must use a supported image extension"
        }
        require(imageBytes.isNotEmpty()) { "imageBytes must not be empty" }

        imagesDir.mkdirs()
        annotationsDir.mkdirs()
        File(imagesDir, imageFileName).writeBytes(imageBytes)

        val annotation = JSONObject()
            .put("schema_version", 1)
            .put("frame_id", frameId)
            .put("image", imageFileName)
            .put("wheels", JSONArray(wheels.map { it.toJson() }))

        File(annotationsDir, "$frameId.json").writeText(annotation.toString(2))
    }

    private fun requireValidFrameId(frameId: String) {
        require(frameId.isNotBlank()) { "frameId must not be blank" }
        require(!frameId.contains(File.separatorChar)) {
            "frameId must be a stem, not a path"
        }
    }

    data class Vec2(val x: Double, val y: Double) {
        init {
            require(x.isFinite() && y.isFinite()) { "Vec2 values must be finite" }
        }

        fun toJsonArray(): JSONArray = JSONArray(listOf(x, y))
    }

    data class BBox(
        val x1: Double,
        val y1: Double,
        val x2: Double,
        val y2: Double
    ) {
        init {
            require(listOf(x1, y1, x2, y2).all { it.isFinite() }) {
                "bbox values must be finite"
            }
            require(x2 > x1 && y2 > y1) {
                "bbox must be [x1, y1, x2, y2] with positive area"
            }
        }

        fun toJsonArray(): JSONArray = JSONArray(listOf(x1, y1, x2, y2))
    }

    data class Points(
        val a: Vec2,
        val b: Vec2,
        val cDiscBottom: Vec2
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("a", a.toJsonArray())
            .put("b", b.toJsonArray())
            .put("c_disc_bottom", cDiscBottom.toJsonArray())
    }

    data class Wheel(
        val bboxXyxy: BBox,
        val points: Points
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("bbox_xyxy", bboxXyxy.toJsonArray())
            .put("points", points.toJson())
    }

    companion object {
        const val DEFAULT_ROOT_NAME = "ar_device_holdout"
        const val SOURCE_TYPE_ANDROID_AR_DEVICE_HUMAN_LABELLED = "android_ar_device_human_labelled"
        const val LABEL_TYPE_HUMAN_REVIEWED = "human_reviewed"
        const val REVIEW_STATUS_ACCEPTED = "accepted"
        val SUPPORTED_IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "bmp", "webp")

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
