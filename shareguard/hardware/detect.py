"""Hardware / runtime detection and ONNX Runtime session factory.

The QNN execution provider ships in the `onnxruntime-qnn` wheel (Windows ARM64).
Provider options follow the ONNX Runtime QNN EP docs: `backend_path` selects
QnnHtp.dll (NPU), QnnGpu.dll or QnnCpu.dll. Verify on the target machine with:

    python -m shareguard.app.main check
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    os: str
    machine: str
    python: str
    processor: str
    ort_version: str | None = None
    providers: list[str] = field(default_factory=list)

    @property
    def is_windows_arm64(self) -> bool:
        return self.os == "Windows" and self.machine.upper() in ("ARM64", "AARCH64")

    @property
    def has_qnn(self) -> bool:
        return "QNNExecutionProvider" in self.providers


def detect() -> HardwareInfo:
    info = HardwareInfo(platform.system(), platform.machine(), platform.python_version(),
                        platform.processor())
    try:
        import onnxruntime as ort
        info.ort_version = ort.__version__
        info.providers = list(ort.get_available_providers())
    except ImportError:
        log.warning("onnxruntime not installed")
    return info


def make_session(model_path: str, device: str = "auto", perf_mode: str = "burst",
                 ep_context_cache: bool = True):
    """Create an ONNX Runtime session on the requested device.

    device: "npu" | "gpu" | "cpu" | "auto" (npu -> gpu -> cpu).
    Returns (session, actual_device). Falls back to CPU instead of crashing.
    """
    import onnxruntime as ort

    info = detect()
    order = {"auto": ["npu", "gpu", "cpu"], "npu": ["npu", "cpu"],
             "gpu": ["gpu", "cpu"], "cpu": ["cpu"]}[device]
    backends = {"npu": "QnnHtp.dll", "gpu": "QnnGpu.dll"}
    for dev in order:
        if dev == "cpu":
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return sess, "cpu"
        if not info.has_qnn:
            continue
        opts = ort.SessionOptions()
        # Fail instead of silently running unsupported nodes on CPU, so a
        # reported "npu" result really is NPU (disable to allow partial offload).
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if ep_context_cache:
            opts.add_session_config_entry("ep.context_enable", "1")
        prov = {"backend_path": backends[dev]}
        if dev == "npu":
            prov["htp_performance_mode"] = perf_mode
        try:
            sess = ort.InferenceSession(model_path, sess_options=opts,
                                        providers=[("QNNExecutionProvider", prov)])
            return sess, dev
        except Exception as e:  # noqa: BLE001 - report and fall through
            log.warning("QNN %s session failed for %s: %s", dev, model_path, e)
    raise RuntimeError("no execution provider could load the model")
