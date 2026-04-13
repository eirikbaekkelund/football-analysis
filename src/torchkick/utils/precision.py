"""GPU precision detection and management."""

from __future__ import annotations

import torch


class PrecisionManager:
    """Auto-detect GPU precision support and provide hints for inference."""

    @staticmethod
    def supports_fp16(device: str = "cuda") -> bool:
        """Check if GPU supports FP16 (compute capability ≥ 7.0)."""
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(device if device == "cuda" else 0)
            return (major, minor) >= (7, 0)
        except Exception:
            return False

    @staticmethod
    def get_optimal_precision(device: str = "cuda", prefer_fp16: bool = True) -> str:
        """Returns 'fp16' if GPU supports it and prefer_fp16=True, else 'fp32'."""
        if prefer_fp16 and PrecisionManager.supports_fp16(device):
            return "fp16"
        return "fp32"

    @staticmethod
    def get_batch_size(vram_gb: float, model_type: str, precision: str) -> int:
        """Estimate batch size based on VRAM and model type."""
        # Approximate memory per crop (in GB) for different models
        memory_per_crop = {
            "siglip": {"fp16": 0.0012, "fp32": 0.0024},
            "pitch": {"fp16": 0.008, "fp32": 0.016},
            "reid": {"fp16": 0.006, "fp32": 0.012},
            "yolo": {"fp16": 0.008, "fp32": 0.016},  # per image, not crop
        }

        per_crop = memory_per_crop.get(model_type, {}).get(precision, 0.01)
        # Reserve 30% of VRAM for overhead (intermediate tensors, etc.)
        available = vram_gb * 0.7
        batch_size = max(8, min(512, int(available / per_crop)))
        return batch_size

    @staticmethod
    def get_available_vram_gb(device: str = "cuda") -> float:
        """Get available GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            total = torch.cuda.get_device_properties(device if device == "cuda" else 0).total_memory
            allocated = torch.cuda.memory_allocated(device if device == "cuda" else 0)
            return (total - allocated) / (1024**3)
        except Exception:
            return 0.0
