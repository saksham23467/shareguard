# Benchmark template (fill only with measured values)

Device: ________  Chip: ________  Windows build: ________  onnxruntime-qnn: ________  Date: ________

| Metric | CPU | GPU (QnnGpu) | NPU (QnnHtp) | Command / source |
|---|---|---|---|---|
| Recogniser latency per call p50 / p95 (ms) | | | | `npu-probe recognizer.onnx` |
| Detector latency per call p50 / p95 (ms) | | | | `npu-probe detector.onnx` |
| Session create / model load (ms) | | | | `npu-probe` |
| Full-scan OCR p50 / p95 (ms) | | | | `demo` |
| Time until safe text readable after window switch p50 / p95 (s) | | | | `demo` |
| Secrets missed / planted | | | | `demo` |
| Pixel-identity violations | | | | `demo` (must be 0) |
| Process RSS (MB) | | | | `bench` |
| Battery drain, 30-min scripted replay (%) | | | | `powercfg /batteryreport` |
| AI Hub hosted-device latency (ms) | — | — | | AI Hub profile job link |
