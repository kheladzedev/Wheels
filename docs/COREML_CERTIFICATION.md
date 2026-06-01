# CoreML Certification

- Certified: True
- Scope: desktop_coreml_package_not_ios_device
- Format: coreml_mlmodel_neuralnetwork
- Artifact: `outputs/production_audit/coreml_export/best.mlmodel`
- Size MB: 37.185
- SHA256: `45ec380a558c59d37cbc664e8464d29cbb8a3bd24e1885f91eca9f99bdbdd0ac`
- Failures: none

## CoreML Spec

- Specification version: 4
- Inputs: `[{'name': 'image', 'kind': 'image', 'width': 640, 'height': 640}]`
- Outputs: `[{'name': 'var_1347', 'kind': 'multiArrayType', 'shape': []}]`

This is desktop package certification, not iOS-device runtime certification.
The iOS team should load the exact `.mlmodel` in the app/XCTest and report latency, memory, and output sanity.
