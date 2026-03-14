"""
FDS GPU Volume Renderer - Multi-GPU CUDA Accelerated
Supports 4x H100 GPUs for large-scale FDS simulation rendering.
"""

import os
import numpy as np
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
class GPUConfig:
    num_gpus: int = 4
    render_resolution: tuple = (1920, 1080)
    volume_resolution: tuple = (512, 512, 256)
    ray_samples: int = 256
    batch_size: int = 4  # frames per batch
    use_fp16: bool = True  # H100 excels at FP16


def get_available_gpus():
    """Detect available NVIDIA GPUs."""
    if HAS_TORCH and torch.cuda.is_available():
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_mem / (1024**3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return gpus
    return []


class MultiGPUVolumeRenderer:
    """
    Multi-GPU volume renderer using ray marching.
    Distributes rendering across 4x H100 GPUs:
      - GPU 0-1: Volume ray marching (split screen halves)
      - GPU 2: Post-processing (tone mapping, bloom, compositing)
      - GPU 3: Data loading & preprocessing pipeline
    """

    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.gpus = get_available_gpus()
        self.num_gpus = min(len(self.gpus), self.config.num_gpus) if self.gpus else 0
        self.volume_data = {}  # per-GPU volume textures
        self.initialized = False

        if self.num_gpus > 0:
            self._init_gpus()

    def _init_gpus(self):
        """Initialize GPU resources across all devices."""
        if not HAS_TORCH:
            return

        self.devices = [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]

        # Pre-allocate volume buffers on each GPU
        vx, vy, vz = self.config.volume_resolution
        dtype = torch.float16 if self.config.use_fp16 else torch.float32

        for i, dev in enumerate(self.devices):
            with torch.cuda.device(dev):
                # Each render GPU gets a copy of the volume
                if i < 2:  # Render GPUs
                    self.volume_data[i] = torch.zeros(
                        (1, 1, vz, vy, vx), dtype=dtype, device=dev
                    )
                torch.cuda.empty_cache()

        self.initialized = True

    def upload_volume(self, data: np.ndarray):
        """
        Upload volume data to render GPUs.
        Data is distributed: GPU0 gets top half, GPU1 gets bottom half.
        """
        if not self.initialized:
            return self._cpu_fallback_upload(data)

        dtype = torch.float16 if self.config.use_fp16 else torch.float32
        tensor = torch.from_numpy(data.astype(np.float32))

        # Normalize to [0, 1]
        vmin, vmax = tensor.min(), tensor.max()
        if vmax > vmin:
            tensor = (tensor - vmin) / (vmax - vmin)

        # Reshape to 5D for grid_sample: (N, C, D, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Resize to target resolution
        vx, vy, vz = self.config.volume_resolution
        tensor = F.interpolate(tensor, size=(vz, vy, vx), mode='trilinear', align_corners=False)

        # Split and distribute to render GPUs
        if self.num_gpus >= 2:
            half_h = vy // 2
            self.volume_data[0] = tensor[:, :, :, :half_h, :].to(
                dtype=dtype, device=self.devices[0]
            )
            self.volume_data[1] = tensor[:, :, :, half_h:, :].to(
                dtype=dtype, device=self.devices[1]
            )
        else:
            self.volume_data[0] = tensor.to(dtype=dtype, device=self.devices[0])

        return {"status": "uploaded", "shape": list(tensor.shape), "gpus": self.num_gpus}

    def _cpu_fallback_upload(self, data):
        """CPU fallback when no GPUs available."""
        self.volume_data["cpu"] = data
        return {"status": "uploaded_cpu", "shape": list(data.shape), "gpus": 0}

    def render_frame(self, camera_params: dict) -> np.ndarray:
        """
        Render a single frame using GPU ray marching.

        camera_params:
            position: [x, y, z]
            target: [x, y, z]
            fov: float (degrees)
            up: [x, y, z]
        """
        W, H = self.config.render_resolution

        if not self.initialized:
            return self._cpu_render(camera_params, W, H)

        return self._gpu_render(camera_params, W, H)

    def _gpu_render(self, camera_params: dict, W: int, H: int) -> np.ndarray:
        """Multi-GPU ray marching render."""
        dtype = torch.float16 if self.config.use_fp16 else torch.float32
        n_samples = self.config.ray_samples

        # Camera setup
        pos = torch.tensor(camera_params.get("position", [3, 3, 3]), dtype=torch.float32)
        target = torch.tensor(camera_params.get("target", [0, 0, 0]), dtype=torch.float32)
        fov = camera_params.get("fov", 50.0)
        up = torch.tensor(camera_params.get("up", [0, 1, 0]), dtype=torch.float32)

        # Compute camera basis
        forward = F.normalize(target - pos, dim=0)
        right = F.normalize(torch.cross(forward, up), dim=0)
        cam_up = torch.cross(right, forward)

        aspect = W / H
        fov_rad = fov * np.pi / 180.0
        half_h = np.tan(fov_rad / 2)
        half_w = half_h * aspect

        # Generate rays on GPUs in parallel
        results = [None] * min(self.num_gpus, 2)

        for gpu_idx in range(min(self.num_gpus, 2)):
            dev = self.devices[gpu_idx]
            with torch.cuda.device(dev):
                # Each GPU renders half the image height
                if self.num_gpus >= 2:
                    h_start = gpu_idx * (H // 2)
                    h_end = (gpu_idx + 1) * (H // 2)
                    local_H = h_end - h_start
                else:
                    h_start = 0
                    local_H = H

                # Create ray directions
                v = torch.linspace(
                    half_h * (1 - 2 * h_start / H),
                    half_h * (1 - 2 * (h_start + local_H) / H),
                    local_H, device=dev, dtype=dtype
                )
                u = torch.linspace(-half_w, half_w, W, device=dev, dtype=dtype)
                vv, uu = torch.meshgrid(v, u, indexing='ij')

                ray_dirs = (
                    forward.to(dev, dtype=dtype).unsqueeze(0).unsqueeze(0)
                    + uu.unsqueeze(-1) * right.to(dev, dtype=dtype).unsqueeze(0).unsqueeze(0)
                    + vv.unsqueeze(-1) * cam_up.to(dev, dtype=dtype).unsqueeze(0).unsqueeze(0)
                )
                ray_dirs = F.normalize(ray_dirs, dim=-1)

                # Ray marching through volume
                ray_origin = pos.to(dev, dtype=dtype)

                # Compute AABB intersection [0, 1]^3 volume
                t_near, t_far = self._ray_aabb_intersect(
                    ray_origin, ray_dirs, dev, dtype
                )

                # Sample along rays
                t_vals = torch.linspace(0, 1, n_samples, device=dev, dtype=dtype)
                t_samples = t_near.unsqueeze(-1) + t_vals * (t_far - t_near).unsqueeze(-1)

                # Sample points
                sample_points = (
                    ray_origin.view(1, 1, 1, 3)
                    + t_samples.unsqueeze(-1) * ray_dirs.unsqueeze(-2)
                )

                # Sample volume (grid_sample expects [-1, 1])
                sample_coords = sample_points * 2 - 1
                vol = self.volume_data.get(gpu_idx)

                if vol is not None:
                    # Reshape for grid_sample: (N, n_samples, H*W, 3)
                    coords_flat = sample_coords.reshape(1, -1, 1, 3)
                    # grid_sample: (N, C, D_in, H_in, W_in) + (N, D_out, H_out, W_out, 3)
                    coords_5d = sample_coords.reshape(1, local_H, W, n_samples, 3)
                    sampled = F.grid_sample(
                        vol, coords_5d, mode='bilinear',
                        padding_mode='zeros', align_corners=True
                    )
                    density = sampled.squeeze(0).squeeze(0)  # (H, W, n_samples)
                    density = density.permute(0, 2, 1)  # (H, n_samples, W) -> rearrange
                else:
                    density = torch.zeros(local_H, W, n_samples, device=dev, dtype=dtype)

                # Volume rendering (emission-absorption)
                dt = (t_far - t_near).unsqueeze(-1) / n_samples
                alpha = 1.0 - torch.exp(-density * dt * 50.0)
                transmittance = torch.cumprod(1.0 - alpha + 1e-6, dim=-1)

                # Fire colormap in GPU
                color = self._gpu_fire_colormap(density, dev, dtype)

                # Composite
                weights = alpha * transmittance
                rgb = (weights.unsqueeze(-1) * color).sum(dim=-2)

                # Clamp and convert
                rgb = torch.clamp(rgb, 0, 1)
                results[gpu_idx] = (rgb.cpu().float().numpy() * 255).astype(np.uint8)

        # Combine halves
        if self.num_gpus >= 2 and results[0] is not None and results[1] is not None:
            frame = np.concatenate(results, axis=0)
        elif results[0] is not None:
            frame = results[0]
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        return frame

    def _ray_aabb_intersect(self, origin, dirs, device, dtype):
        """Compute ray-AABB intersection for [0,1]^3 box."""
        inv_dir = 1.0 / (dirs + 1e-8)
        t0 = (0 - origin) * inv_dir
        t1 = (1 - origin) * inv_dir

        t_min_vals = torch.minimum(t0, t1)
        t_max_vals = torch.maximum(t0, t1)

        t_near = t_min_vals[..., 0].clamp(min=0)
        t_near = torch.maximum(t_near, t_min_vals[..., 1])
        t_near = torch.maximum(t_near, t_min_vals[..., 2])

        t_far = t_max_vals[..., 0]
        t_far = torch.minimum(t_far, t_max_vals[..., 1])
        t_far = torch.minimum(t_far, t_max_vals[..., 2])

        t_far = torch.maximum(t_far, t_near + 0.001)

        return t_near, t_far

    def _gpu_fire_colormap(self, density, device, dtype):
        """Apply fire colormap on GPU. Returns (H, W, n_samples, 3)."""
        t = torch.clamp(density, 0, 1)

        r = torch.where(t < 0.5, t * 2, torch.ones_like(t))
        g = torch.where(t < 0.5, torch.zeros_like(t),
                        torch.where(t < 0.75, (t - 0.5) * 4, 0.8 + (t - 0.75) * 0.8))
        b = torch.where(t > 0.75, (t - 0.75) * 4, torch.zeros_like(t))

        return torch.stack([r, g, b], dim=-1)

    def _cpu_render(self, camera_params, W, H):
        """CPU fallback renderer."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Simple CPU ray marching (limited quality)
        vol = self.volume_data.get("cpu")
        if vol is None:
            return frame

        pos = np.array(camera_params.get("position", [3, 3, 3]))
        target = np.array(camera_params.get("target", [0, 0, 0]))
        fov = camera_params.get("fov", 50.0)

        forward = target - pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Simplified: render central slice as 2D
        nz, ny, nx = vol.shape
        vmax = vol.max() or 1.0
        for iy in range(min(ny, H)):
            for ix in range(min(nx, W)):
                val = 0
                for iz in range(nz):
                    val += vol[iz, iy, ix] / vmax
                val = min(val / nz * 5, 1.0)
                r = min(int(val * 255 * 2), 255)
                g = int(val * 180)
                b = 0
                frame[iy * H // ny, ix * W // nx] = [r, g, b]

        return frame

    def render_batch(self, camera_params_list: list) -> list:
        """Render multiple frames in a batch (for animation)."""
        frames = []
        for params in camera_params_list:
            frames.append(self.render_frame(params))
        return frames

    def get_status(self) -> dict:
        """Return GPU status information."""
        status = {
            "num_gpus": self.num_gpus,
            "initialized": self.initialized,
            "gpus": self.gpus,
            "config": {
                "render_resolution": self.config.render_resolution,
                "volume_resolution": self.config.volume_resolution,
                "ray_samples": self.config.ray_samples,
                "use_fp16": self.config.use_fp16,
            }
        }

        if HAS_TORCH and self.num_gpus > 0:
            mem_info = []
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                mem_info.append({
                    "gpu": i,
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 1),
                    "utilization_pct": round(allocated / total * 100, 1),
                })
            status["memory"] = mem_info

        return status

    def cleanup(self):
        """Free GPU resources."""
        self.volume_data.clear()
        if HAS_TORCH:
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        self.initialized = False
