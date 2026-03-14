"""
FDS GPU Data Pipeline - Multi-GPU Data Loading & Preprocessing
Uses GPU 3 (dedicated) for async data loading and preprocessing.
"""

import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class PipelineConfig:
    preprocess_gpu: int = 3       # Dedicated GPU for data loading
    prefetch_frames: int = 8      # Frames to prefetch
    max_cache_gb: float = 20.0    # Max GPU cache size in GB
    compress_ratio: float = 0.5   # Volume downscale for preview
    use_fp16: bool = True


class GPUDataPipeline:
    """
    Async GPU data pipeline for FDS simulation data.

    Architecture (4x H100):
      GPU 0: Render (top half)
      GPU 1: Render (bottom half)
      GPU 2: Post-processing & compositing
      GPU 3: Data loading, preprocessing, prefetching

    This class manages GPU 3 for data operations.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.frame_cache = {}
        self.cache_order = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._setup_device()

    def _setup_device(self):
        """Initialize the preprocessing GPU."""
        if HAS_TORCH and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_idx = min(self.config.preprocess_gpu, gpu_count - 1)
            self.device = torch.device(f"cuda:{gpu_idx}")
            self.gpu_available = True
        else:
            self.device = torch.device("cpu") if HAS_TORCH else None
            self.gpu_available = False

    def preprocess_volume(self, raw_data: np.ndarray, target_shape: tuple = None) -> dict:
        """
        Preprocess raw FDS volume data on GPU.

        Steps:
        1. Upload to GPU 3
        2. Normalize values
        3. Apply Gaussian smoothing
        4. Resize to target resolution
        5. Compute gradient (for lighting)
        6. Generate multi-resolution LODs
        """
        if not HAS_TORCH:
            return self._cpu_preprocess(raw_data, target_shape)

        with torch.cuda.device(self.device):
            # Upload
            tensor = torch.from_numpy(raw_data.astype(np.float32)).to(self.device)

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

            # Normalize
            vmin, vmax = tensor.min(), tensor.max()
            if vmax > vmin:
                tensor = (tensor - vmin) / (vmax - vmin)

            # Gaussian smoothing (3x3x3 kernel)
            kernel_size = 3
            sigma = 0.8
            coords = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
            kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            kernel_3d = kernel_3d / kernel_3d.sum()
            kernel_3d = kernel_3d.reshape(1, 1, kernel_size, kernel_size, kernel_size)

            smoothed = F.conv3d(tensor, kernel_3d, padding=kernel_size // 2)

            # Resize to target
            if target_shape:
                smoothed = F.interpolate(
                    smoothed, size=target_shape, mode='trilinear', align_corners=False
                )

            # Compute gradient for lighting normals
            grad_x = smoothed[:, :, :, :, 1:] - smoothed[:, :, :, :, :-1]
            grad_y = smoothed[:, :, :, 1:, :] - smoothed[:, :, :, :-1, :]
            grad_z = smoothed[:, :, 1:, :, :] - smoothed[:, :, :-1, :, :]

            # Generate LOD pyramid
            lods = []
            current = smoothed
            for level in range(4):
                shape = current.shape[2:]
                lods.append({
                    "level": level,
                    "shape": list(shape),
                    "data": current.cpu().numpy().squeeze(),
                    "max_val": float(current.max()),
                })
                if min(shape) > 8:
                    current = F.avg_pool3d(current, 2)

            dtype = torch.float16 if self.config.use_fp16 else torch.float32

            result = {
                "volume": smoothed.to(dtype).cpu().numpy().squeeze(),
                "gradient_magnitude": (
                    grad_x[:, :, :, :, :grad_x.shape[-1]].abs() +
                    grad_y[:, :, :, :grad_y.shape[-2], :].abs() +
                    grad_z[:, :, :grad_z.shape[-3], :, :].abs()
                ).cpu().numpy().squeeze() if grad_x.numel() > 0 else None,
                "lods": lods,
                "original_range": [float(vmin), float(vmax)],
                "shape": list(smoothed.shape[2:]),
            }

            torch.cuda.empty_cache()
            return result

    def _cpu_preprocess(self, raw_data, target_shape):
        """CPU fallback preprocessing."""
        data = raw_data.astype(np.float32)
        vmin, vmax = data.min(), data.max()
        if vmax > vmin:
            data = (data - vmin) / (vmax - vmin)
        return {
            "volume": data,
            "gradient_magnitude": None,
            "lods": [{"level": 0, "shape": list(data.shape), "data": data, "max_val": float(data.max())}],
            "original_range": [float(vmin), float(vmax)],
            "shape": list(data.shape),
        }

    def cache_frame(self, frame_id: str, data: dict):
        """Cache a preprocessed frame on GPU."""
        self.frame_cache[frame_id] = data
        self.cache_order.append(frame_id)

        # Evict old frames if cache too large
        while len(self.frame_cache) > self.config.prefetch_frames * 4:
            oldest = self.cache_order.pop(0)
            del self.frame_cache[oldest]

    def get_cached_frame(self, frame_id: str):
        """Retrieve a cached frame."""
        return self.frame_cache.get(frame_id)

    def prefetch_frames(self, frame_ids: list, data_loader_fn):
        """
        Prefetch upcoming frames in background.
        Uses ThreadPoolExecutor for async I/O + GPU preprocessing.
        """
        futures = []
        for fid in frame_ids:
            if fid not in self.frame_cache:
                future = self.executor.submit(self._prefetch_single, fid, data_loader_fn)
                futures.append(future)
        return futures

    def _prefetch_single(self, frame_id, loader_fn):
        """Load and preprocess a single frame."""
        raw = loader_fn(frame_id)
        if raw is not None:
            processed = self.preprocess_volume(raw)
            self.cache_frame(frame_id, processed)
            return frame_id
        return None

    def get_memory_stats(self) -> dict:
        """Return GPU memory statistics."""
        stats = {
            "cached_frames": len(self.frame_cache),
            "gpu_available": self.gpu_available,
        }
        if self.gpu_available and HAS_TORCH:
            idx = self.device.index or 0
            stats["gpu_index"] = idx
            stats["allocated_gb"] = round(torch.cuda.memory_allocated(idx) / (1024**3), 2)
            stats["reserved_gb"] = round(torch.cuda.memory_reserved(idx) / (1024**3), 2)
            stats["total_gb"] = round(
                torch.cuda.get_device_properties(idx).total_mem / (1024**3), 1
            )
        return stats

    def cleanup(self):
        """Release all GPU resources."""
        self.frame_cache.clear()
        self.cache_order.clear()
        self.executor.shutdown(wait=False)
        if self.gpu_available and HAS_TORCH:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
