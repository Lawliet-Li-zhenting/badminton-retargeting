"""Vision-based badminton motion retargeting and real-time match."""
import os as _os

# Laptops with two GPUs give an OpenGL context of python.exe to the integrated one by default
# (measured here: 49 ms per 1280x720 frame on the Intel iGPU, 5 ms on the RTX 4060 next to it).
# This is the documented per-process switch of NVIDIA's Optimus layer; it has to be in the
# environment before the first GL context is created and is ignored on other machines.
# BADRT_GPU=integrated keeps the default.
if _os.name == "nt" and _os.environ.get("BADRT_GPU", "").lower() != "integrated":
    _os.environ.setdefault("SHIM_MCCOMPAT", "0x800000001")
