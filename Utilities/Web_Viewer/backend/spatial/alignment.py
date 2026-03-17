"""
Spatial Alignment
Aligns 3D scan data with FDS simulation domain.
Supports manual transform, ICP registration, and marker-based alignment.
"""

import numpy as np
from dataclasses import dataclass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SpatialTransform:
    """Rigid body transform: rotation + translation + scale."""
    translation: np.ndarray = None  # (3,)
    rotation: np.ndarray = None  # (3, 3) rotation matrix
    scale: float = 1.0

    def __post_init__(self):
        if self.translation is None:
            self.translation = np.zeros(3, dtype=np.float64)
        if self.rotation is None:
            self.rotation = np.eye(3, dtype=np.float64)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to (N, 3) points."""
        return (points @ self.rotation.T) * self.scale + self.translation

    def to_matrix4x4(self) -> list:
        """Convert to column-major 4x4 matrix for Three.js."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation * self.scale
        m[:3, 3] = self.translation
        # Column-major for Three.js
        return m.T.flatten().tolist()

    def to_dict(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "scale": self.scale,
            "matrix4x4": self.to_matrix4x4(),
        }

    @staticmethod
    def from_dict(d: dict) -> 'SpatialTransform':
        return SpatialTransform(
            translation=np.array(d.get("translation", [0, 0, 0]), dtype=np.float64),
            rotation=np.array(d.get("rotation", np.eye(3).tolist()), dtype=np.float64),
            scale=d.get("scale", 1.0),
        )


def compute_auto_alignment(
    scan_bounds_min: np.ndarray,
    scan_bounds_max: np.ndarray,
    fds_bounds_min: np.ndarray,
    fds_bounds_max: np.ndarray,
) -> SpatialTransform:
    """
    Compute transform to align scan bounding box to FDS domain.
    Scales and translates scan to fit within FDS bounds.
    """
    scan_center = (scan_bounds_min + scan_bounds_max) / 2
    scan_size = scan_bounds_max - scan_bounds_min
    fds_center = (fds_bounds_min + fds_bounds_max) / 2
    fds_size = fds_bounds_max - fds_bounds_min

    # Scale to fit
    scale_factors = fds_size / (scan_size + 1e-8)
    scale = float(np.min(scale_factors)) * 0.95  # 95% to leave margin

    # Translation: align centers
    translation = fds_center - scan_center * scale

    return SpatialTransform(
        translation=translation,
        rotation=np.eye(3, dtype=np.float64),
        scale=scale,
    )


def compute_marker_alignment(
    scan_markers: np.ndarray,
    fds_markers: np.ndarray,
) -> SpatialTransform:
    """
    Compute optimal rigid transform from marker correspondences.
    Uses SVD-based Procrustes alignment.
    Requires at least 3 corresponding point pairs.

    scan_markers: (N, 3) points in scan space
    fds_markers: (N, 3) corresponding points in FDS space
    """
    assert len(scan_markers) >= 3, "Need at least 3 marker pairs"
    assert len(scan_markers) == len(fds_markers)

    scan_pts = np.array(scan_markers, dtype=np.float64)
    fds_pts = np.array(fds_markers, dtype=np.float64)

    # Centroids
    scan_centroid = scan_pts.mean(axis=0)
    fds_centroid = fds_pts.mean(axis=0)

    # Center
    scan_centered = scan_pts - scan_centroid
    fds_centered = fds_pts - fds_centroid

    # Scale
    scan_scale = np.sqrt((scan_centered ** 2).sum() / len(scan_pts))
    fds_scale = np.sqrt((fds_centered ** 2).sum() / len(fds_pts))
    scale = fds_scale / (scan_scale + 1e-8)

    # Rotation via SVD
    H = scan_centered.T @ fds_centered
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Translation
    t = fds_centroid - (scan_centroid @ R.T) * scale

    return SpatialTransform(
        translation=t,
        rotation=R,
        scale=scale,
    )


def compute_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_points: int = 10000,
) -> SpatialTransform:
    """
    Iterative Closest Point (ICP) registration.
    Aligns source points to target points.
    """
    # Downsample for performance
    if len(source) > max_points:
        idx = np.random.choice(len(source), max_points, replace=False)
        src = source[idx].copy()
    else:
        src = source.copy()

    if len(target) > max_points:
        idx = np.random.choice(len(target), max_points, replace=False)
        tgt = target[idx]
    else:
        tgt = target

    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_error = float('inf')

    # Build KD-tree for fast nearest-neighbor queries (O(n log n) vs O(n²))
    if HAS_SCIPY:
        tree = cKDTree(tgt)

    for iteration in range(max_iterations):
        # Find closest points using KD-tree (O(n log m)) or brute-force fallback
        if HAS_SCIPY:
            _, correspondences = tree.query(src, k=1)
        else:
            correspondences = np.zeros(len(src), dtype=np.int32)
            for i, p in enumerate(src):
                dists = np.sum((tgt - p) ** 2, axis=1)
                correspondences[i] = np.argmin(dists)

        matched_target = tgt[correspondences]

        # Compute transform
        src_centroid = src.mean(axis=0)
        tgt_centroid = matched_target.mean(axis=0)

        src_centered = src - src_centroid
        tgt_centered = matched_target - tgt_centroid

        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        t = tgt_centroid - src_centroid @ R.T

        # Apply
        src = (src @ R.T) + t
        R_total = R @ R_total
        t_total = t + t_total @ R.T

        # Check convergence
        error = np.mean(np.sum((src - matched_target) ** 2, axis=1))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return SpatialTransform(
        translation=t_total,
        rotation=R_total,
        scale=1.0,
    )
