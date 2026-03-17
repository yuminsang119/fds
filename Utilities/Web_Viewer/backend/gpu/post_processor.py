"""
GPU Post-Processor - Runs on GPU 2
Tone mapping, bloom, SSAO, and frame compositing.
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GPUPostProcessor:
    """
    Post-processing pipeline on GPU 2.
    Applies cinematic effects to rendered frames.
    """

    def __init__(self, gpu_index: int = 2):
        if HAS_TORCH and torch.cuda.is_available():
            idx = min(gpu_index, torch.cuda.device_count() - 1)
            self.device = torch.device(f"cuda:{idx}")
            self.gpu_available = True
        else:
            self.device = None
            self.gpu_available = False

        self.bloom_kernel = None
        self.exposure = 1.0
        self.gamma = 2.2
        self.bloom_intensity = 0.3
        self.bloom_threshold = 0.7

    def process(self, frame: np.ndarray, effects: dict = None) -> np.ndarray:
        """
        Apply post-processing effects to a rendered frame.

        effects:
            tone_mapping: bool (ACES tone mapping)
            bloom: bool (glow on bright areas)
            gamma: float
            exposure: float
        """
        if effects is None:
            effects = {"tone_mapping": True, "bloom": True}

        if not self.gpu_available or not HAS_TORCH:
            return self._cpu_process(frame, effects)

        with torch.cuda.device(self.device):
            # Upload frame: (H, W, 3) -> (1, 3, H, W)
            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

            # Exposure
            exposure = effects.get("exposure", self.exposure)
            tensor = tensor * exposure

            # Bloom
            if effects.get("bloom", False):
                tensor = self._apply_bloom(tensor)

            # Tone mapping (ACES filmic)
            if effects.get("tone_mapping", True):
                tensor = self._aces_tonemap(tensor)

            # Gamma correction
            gamma = effects.get("gamma", self.gamma)
            tensor = torch.pow(torch.clamp(tensor, 0, 1), 1.0 / gamma)

            # Convert back
            result = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return result

    def _aces_tonemap(self, x):
        """ACES filmic tone mapping curve."""
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    def _apply_bloom(self, tensor):
        """GPU-accelerated bloom with multi-scale downsampled passes."""
        threshold = self.bloom_threshold
        bright = torch.clamp(tensor - threshold, 0, 1)

        # Multi-scale bloom: blur at progressively lower resolutions
        blurred = bright
        bloom_accumulator = torch.zeros_like(tensor)
        weights = [0.5, 0.3, 0.2]
        for i, w in enumerate(weights):
            if i > 0:
                # Downsample by 2x for each level (4x, 16x fewer pixels)
                blurred = F.avg_pool2d(blurred, 2)
            blurred = self._gaussian_blur(blurred, kernel_size=9, sigma=2.0)
            # Upsample back to original resolution
            upsampled = F.interpolate(blurred, size=tensor.shape[2:], mode='bilinear', align_corners=False)
            bloom_accumulator += upsampled * w

        return tensor + bloom_accumulator * self.bloom_intensity

    def _gaussian_blur(self, tensor, kernel_size=15, sigma=3.0):
        """2D Gaussian blur on GPU."""
        if self.bloom_kernel is None or self.bloom_kernel.shape[-1] != kernel_size:
            coords = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
            kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()

            # Separable: horizontal then vertical
            self.bloom_kernel = kernel_1d

        k = self.bloom_kernel
        C = tensor.shape[1]
        pad = kernel_size // 2

        # Horizontal pass
        k_h = k.reshape(1, 1, 1, -1).expand(C, 1, 1, -1)
        out = F.conv2d(tensor, k_h, padding=(0, pad), groups=C)

        # Vertical pass
        k_v = k.reshape(1, 1, -1, 1).expand(C, 1, -1, 1)
        out = F.conv2d(out, k_v, padding=(pad, 0), groups=C)

        return out

    def _cpu_process(self, frame, effects):
        """Simple CPU fallback."""
        result = frame.astype(np.float32) / 255.0

        exposure = effects.get("exposure", 1.0)
        result *= exposure

        gamma = effects.get("gamma", 2.2)
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / gamma)

        return (result * 255).astype(np.uint8)

    def cleanup(self):
        """Free GPU resources."""
        self.bloom_kernel = None
        if self.gpu_available and HAS_TORCH:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
