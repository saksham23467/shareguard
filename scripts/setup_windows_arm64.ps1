# ShareGuard setup for Snapdragon (Windows on ARM). UNTESTED on real hardware as of submission.
# Requires Python 3.11/3.12 ARM64 from python.org.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Swap CPU onnxruntime for the QNN build (same `import onnxruntime`, adds QNNExecutionProvider).
pip uninstall -y onnxruntime
pip install onnxruntime-qnn
python -m shareguard.app.main check     # expect "QNNExecutionProvider" in providers
python -m pytest -q                     # expect all tests passing
