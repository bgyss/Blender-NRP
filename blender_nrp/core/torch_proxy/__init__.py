"""Optional-dependency PyTorch neural render proxy (nrp torch_backend parity).

Everything in this package imports torch lazily at module level *inside the
submodules*; nothing under ``blender_nrp.core`` imports this package implicitly, so
the base add-on stays numpy-only. Call `torch_status()` first: it reports whether
torch is importable without raising, which is what the operators use to degrade
gracefully.
"""

from __future__ import annotations


def torch_status() -> tuple[bool, str]:
    """(available, human-readable detail). Never raises."""
    try:
        import torch
    except Exception as exc:  # ImportError or a broken install
        return False, (
            f"PyTorch is not available ({exc}). Install torch into Blender's Python "
            "to enable proxy training/inference; cache-gather preview works without it."
        )
    return True, f"torch {torch.__version__}"


def select_device(preference: str = "auto") -> str:
    """Resolve 'auto' to mps/cuda/cpu based on availability."""
    import torch

    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
