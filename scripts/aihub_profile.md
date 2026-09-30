# Measuring the NPU on Qualcomm AI Hub (hosted Snapdragon devices)

These commands use the public `qai-hub` / `qai-hub-models` packages. Flags change between
releases, so check `--help` first rather than trusting this file.

```bash
pip install qai-hub "qai-hub-models[easyocr]"
qai-hub configure --api_token $QAIHUB_API_TOKEN
python -m qai_hub_models.models.easyocr.export --help      # verify flags first
python -m qai_hub_models.models.easyocr.export --device "Snapdragon X Elite CRD" --target-runtime onnx
```

The export prints AI Hub profile-job links with on-device latency. Copy those numbers into
`docs/BENCHMARKS.md` as "AI Hub hosted X Elite CRD", not as HP laptop numbers.
The downloaded ONNX files can then be timed on a real Snapdragon laptop with:

```bash
python -m shareguard.app.main npu-probe path/to/model.onnx --devices npu gpu cpu
```
