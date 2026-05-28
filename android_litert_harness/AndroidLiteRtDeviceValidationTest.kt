package com.vsbl.wheels.ml

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.LocalDate
import java.time.ZoneOffset
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

@RunWith(AndroidJUnit4::class)
class AndroidLiteRtDeviceValidationTest {
    @Test
    fun produceAndroidLiteRtDeviceReport() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelBytes = context.assets.open(MODEL_ASSET_NAME).use { it.readBytes() }
        val modelSha256 = sha256(modelBytes)
        val expectedArtifactSha256 = expectedArtifactSha256(context)
        assertEquals(
            "Android test asset does not match the ML-provided expected TFLite artifact",
            expectedArtifactSha256,
            modelSha256
        )
        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes)
        modelBuffer.rewind()

        Interpreter(modelBuffer).use { interpreter ->
            val inputShape = interpreter.getInputTensor(0).shape()
            val inputDtype = interpreter.getInputTensor(0).dataType().name.lowercase()
            val outputShape = interpreter.getOutputTensor(0).shape()
            assertArrayEquals(EXPECTED_INPUT_SHAPE, inputShape)
            assertArrayEquals(EXPECTED_OUTPUT_SHAPE, outputShape)
            require(inputDtype == EXPECTED_INPUT_DTYPE) {
                "LiteRT input dtype must be $EXPECTED_INPUT_DTYPE for production evidence"
            }

            val inputBuffer = zeroFloatBuffer(inputShape.product())
            val outputBuffer = ByteBuffer
                .allocateDirect(outputShape.product() * FLOAT_BYTES)
                .order(ByteOrder.nativeOrder())

            repeat(WARMUP_RUNS) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter.run(inputBuffer, outputBuffer)
            }

            val latenciesMs = mutableListOf<Double>()
            repeat(MEASURED_RUNS) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                val startNs = SystemClock.elapsedRealtimeNanos()
                interpreter.run(inputBuffer, outputBuffer)
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
                latenciesMs.add(elapsedMs)
            }

            outputBuffer.rewind()
            val outputStats = outputStats(outputBuffer, outputShape.product())
            assertTrue("LiteRT output contains non-finite values", outputStats.finite)
            require(outputStats.min < outputStats.max) {
                "LiteRT output range must be non-degenerate for production evidence"
            }
            require(outputStats.mean >= outputStats.min && outputStats.mean <= outputStats.max) {
                "LiteRT output mean must lie within [min, max]"
            }
            val likelyEmulator = isLikelyEmulator()
            require(!likelyEmulator) {
                "Android LiteRT production evidence must run on a physical device"
            }
            val meanLatencyMs = latenciesMs.average()
            val p95LatencyMs = percentile(latenciesMs, 0.95)
            val peakMemoryMb = currentPssMb()
            require(peakMemoryMb > 0.0) {
                "peak memory must be positive for production evidence"
            }

            val report = JSONObject()
                .put("schema_version", 1)
                .put("source_type", SOURCE_TYPE_ANDROID_LITERT_DEVICE_VALIDATION)
                .put("test_session_id", "android_litert_${System.currentTimeMillis()}")
                .put("test_app_version", appVersion(context))
                .put("test_date_utc", LocalDate.now(ZoneOffset.UTC).toString())
                .put(
                    "device",
                    JSONObject()
                        .put("model", Build.MODEL ?: "")
                        .put("manufacturer", Build.MANUFACTURER ?: "")
                        .put("android_version", Build.VERSION.RELEASE ?: "")
                        .put("sdk_int", Build.VERSION.SDK_INT)
                        .put("soc", Build.HARDWARE ?: "")
                        .put("is_emulator", likelyEmulator)
                )
                .put("runtime", "LiteRT")
                .put(
                    "artifact",
                    JSONObject()
                        .put("path", "outputs/production_audit/tflite_export/best_float32.tflite")
                        .put("sha256", modelSha256)
                        .put("expected_sha256", expectedArtifactSha256)
                        .put("format", "tflite_float32")
                )
                .put(
                    "input",
                    JSONObject()
                        .put("shape", JSONArray(inputShape.toList()))
                        .put("dtype", inputDtype)
                        .put("profile", INPUT_PROFILE_ZERO_FLOAT32_SMOKE)
                )
                .put(
                    "latency_ms",
                    JSONObject()
                        .put("runs", latenciesMs.size)
                        .put("mean", meanLatencyMs)
                        .put("p50", percentile(latenciesMs, 0.50))
                        .put("p95", p95LatencyMs)
                        .put("stdev", sampleStdev(latenciesMs))
                )
                .put(
                    "memory_mb",
                    JSONObject()
                        .put("peak", peakMemoryMb)
                )
                .put(
                    "output",
                    JSONObject()
                        .put("shape", JSONArray(outputShape.toList()))
                        .put("finite", outputStats.finite)
                        .put("min", outputStats.min)
                        .put("max", outputStats.max)
                        .put("mean", outputStats.mean)
                )

            val outFile = File(context.requireExternalFilesDir(), REPORT_FILE_NAME)
            outFile.writeText(report.toString(2))
            println("Android LiteRT device report: ${outFile.absolutePath}")
        }
    }

    private fun zeroFloatBuffer(floatCount: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(floatCount * FLOAT_BYTES).order(ByteOrder.nativeOrder())
        repeat(floatCount) { buffer.putFloat(0f) }
        buffer.rewind()
        return buffer
    }

    private fun outputStats(buffer: ByteBuffer, floatCount: Int): OutputStats {
        val floats = buffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
        var finite = true
        var min = Float.POSITIVE_INFINITY
        var max = Float.NEGATIVE_INFINITY
        var sum = 0.0
        repeat(floatCount) {
            val value = floats.get()
            if (!value.isFinite()) {
                finite = false
            }
            if (value < min) {
                min = value
            }
            if (value > max) {
                max = value
            }
            sum += value
        }
        return OutputStats(
            finite = finite,
            min = min.toDouble(),
            max = max.toDouble(),
            mean = sum / floatCount
        )
    }

    private fun percentile(values: List<Double>, q: Double): Double {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        if (sorted.size == 1) {
            return sorted.first()
        }
        val pos = (sorted.size - 1) * q
        val lo = kotlin.math.floor(pos).toInt().coerceIn(0, sorted.lastIndex)
        val hi = kotlin.math.ceil(pos).toInt().coerceIn(0, sorted.lastIndex)
        val frac = pos - lo
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    private fun sampleStdev(values: List<Double>): Double {
        if (values.size < 2) {
            return 0.0
        }
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / (values.size - 1)
        return sqrt(variance)
    }

    private fun currentPssMb(): Double {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss / 1024.0
    }

    private fun sha256(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun isLikelyEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT.orEmpty().lowercase()
        val model = Build.MODEL.orEmpty().lowercase()
        val manufacturer = Build.MANUFACTURER.orEmpty().lowercase()
        val brand = Build.BRAND.orEmpty().lowercase()
        val device = Build.DEVICE.orEmpty().lowercase()
        val product = Build.PRODUCT.orEmpty().lowercase()
        val hardware = Build.HARDWARE.orEmpty().lowercase()
        return fingerprint.startsWith("generic") ||
            fingerprint.contains("emulator") ||
            model.contains("google_sdk") ||
            model.contains("emulator") ||
            model.contains("android sdk built for") ||
            manufacturer.contains("genymotion") ||
            hardware in setOf("goldfish", "ranchu") ||
            (brand.startsWith("generic") && device.startsWith("generic")) ||
            product.contains("sdk")
    }

    private fun expectedArtifactSha256(context: Context): String {
        val json = context.assets.open(EXPECTED_ARTIFACT_ASSET_NAME).bufferedReader().use {
            it.readText()
        }
        val payload = JSONObject(json)
        val sha = payload
            .getJSONObject("expected_android_artifact")
            .getString("sha256")
            .trim()
        require(sha.isNotBlank()) {
            "EXPECTED_ANDROID_ARTIFACT.json must include expected_android_artifact.sha256"
        }
        return sha
    }

    private fun appVersion(context: Context): String {
        val version = context.packageManager
            .getPackageInfo(context.packageName, 0)
            .versionName
            .orEmpty()
            .trim()
        require(version.isNotBlank()) {
            "Android package versionName must be non-blank for production LiteRT evidence"
        }
        return version
    }

    private fun Context.requireExternalFilesDir(): File {
        return requireNotNull(getExternalFilesDir(null)) {
            "External files directory is unavailable"
        }
    }

    private fun IntArray.product(): Int = fold(1) { acc, value -> acc * value }

    private data class OutputStats(
        val finite: Boolean,
        val min: Double,
        val max: Double,
        val mean: Double
    )

    private companion object {
        private const val MODEL_ASSET_NAME = "best_float32.tflite"
        private const val EXPECTED_ARTIFACT_ASSET_NAME = "EXPECTED_ANDROID_ARTIFACT.json"
        private const val REPORT_FILE_NAME = "android_litert_device_report.json"
        private const val SOURCE_TYPE_ANDROID_LITERT_DEVICE_VALIDATION =
            "android_litert_device_validation"
        private const val FLOAT_BYTES = 4
        private const val WARMUP_RUNS = 5
        private const val MEASURED_RUNS = 30
        private const val INPUT_PROFILE_ZERO_FLOAT32_SMOKE = "zero_float32_smoke"
        private const val EXPECTED_INPUT_DTYPE = "float32"
        private val EXPECTED_INPUT_SHAPE = intArrayOf(1, 640, 640, 3)
        private val EXPECTED_OUTPUT_SHAPE = intArrayOf(1, 14, 8400)
    }
}
