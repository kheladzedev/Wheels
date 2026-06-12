import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const [modelArg, manifestArg, reportArg] = process.argv.slice(2);
if (!modelArg || !manifestArg || !reportArg) {
  console.error("usage: node smoke_onnxruntime_web.mjs <model.onnx> <manifest.json> <report.json>");
  process.exit(2);
}

const modelPath = path.resolve(__dirname, modelArg);
const manifestPath = path.resolve(__dirname, manifestArg);
const reportPath = path.resolve(__dirname, reportArg);
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const wasmDir = path.resolve(__dirname, "node_modules/onnxruntime-web/dist") + path.sep;
ort.env.wasm.wasmPaths = wasmDir;

const modelBytes = fs.readFileSync(modelPath);
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ["wasm"],
});
const inputShape = manifest.input_shape;
const n = inputShape.reduce((acc, value) => acc * value, 1);
const image = new ort.Tensor("float32", new Float32Array(n), inputShape);
const results = await session.run({ image });
const outputs = {};
for (const [name, tensor] of Object.entries(results)) {
  outputs[name] = {
    dims: Array.from(tensor.dims),
    type: tensor.type,
    size: tensor.data.length,
  };
}
const report = {
  ok: true,
  executionProvider: "wasm",
  inputName: "image",
  inputShape,
  outputNames: Object.keys(results).sort(),
  outputs,
  runtimeScope: manifest.runtime_scope,
};
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
