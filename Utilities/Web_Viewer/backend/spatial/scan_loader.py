"""
3D Scan Data Loader
Supports PLY (point cloud/mesh), GLTF/GLB (photogrammetry mesh),
LAS/LAZ (LiDAR), and OBJ (mesh) formats.
"""

import struct
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PointCloud:
    """3D point cloud with optional colors and normals."""
    points: np.ndarray  # (N, 3) xyz
    colors: np.ndarray = None  # (N, 3) rgb [0-255]
    normals: np.ndarray = None  # (N, 3)
    intensity: np.ndarray = None  # (N,) for LiDAR
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.ones(3))

    def compute_bounds(self):
        self.bounds_min = self.points.min(axis=0)
        self.bounds_max = self.points.max(axis=0)

    def downsample(self, voxel_size: float):
        """Voxel-based downsampling for large point clouds."""
        shifted = self.points - self.bounds_min
        voxel_indices = (shifted / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(
            voxel_indices, axis=0, return_index=True
        )
        self.points = self.points[unique_idx]
        if self.colors is not None:
            self.colors = self.colors[unique_idx]
        if self.normals is not None:
            self.normals = self.normals[unique_idx]
        if self.intensity is not None:
            self.intensity = self.intensity[unique_idx]
        self.compute_bounds()
        return len(unique_idx)

    def to_json(self, max_points: int = 500000) -> dict:
        """Convert to JSON-serializable dict, downsampling if needed."""
        pts = self.points
        cols = self.colors
        norms = self.normals

        if len(pts) > max_points:
            indices = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[indices]
            if cols is not None:
                cols = cols[indices]
            if norms is not None:
                norms = norms[indices]

        result = {
            "num_points": len(pts),
            "positions": pts.flatten().tolist(),
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
        }
        if cols is not None:
            result["colors"] = (cols / 255.0).flatten().tolist()
        if norms is not None:
            result["normals"] = norms.flatten().tolist()
        return result


@dataclass
class TriMesh:
    """Triangle mesh with optional texture."""
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) indices
    normals: np.ndarray = None  # (N, 3)
    uvs: np.ndarray = None  # (N, 2)
    vertex_colors: np.ndarray = None  # (N, 3)
    texture_data: bytes = None
    texture_width: int = 0
    texture_height: int = 0
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.ones(3))

    def compute_bounds(self):
        self.bounds_min = self.vertices.min(axis=0)
        self.bounds_max = self.vertices.max(axis=0)

    def compute_normals(self):
        """Compute per-vertex normals from face normals (vectorized)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        face_normals /= norms
        self.normals = np.zeros_like(self.vertices)
        np.add.at(self.normals, self.faces[:, 0], face_normals)
        np.add.at(self.normals, self.faces[:, 1], face_normals)
        np.add.at(self.normals, self.faces[:, 2], face_normals)
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normals /= norms

    def simplify(self, target_faces: int):
        """Simple mesh decimation by random face removal."""
        if len(self.faces) <= target_faces:
            return
        indices = np.random.choice(len(self.faces), target_faces, replace=False)
        self.faces = self.faces[indices]

    def to_json(self, max_faces: int = 200000) -> dict:
        """Convert to JSON-serializable dict."""
        faces = self.faces
        if len(faces) > max_faces:
            indices = np.random.choice(len(faces), max_faces, replace=False)
            faces = faces[indices]

        result = {
            "num_vertices": len(self.vertices),
            "num_faces": len(faces),
            "positions": self.vertices.flatten().tolist(),
            "indices": faces.flatten().tolist(),
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
        }
        if self.normals is not None:
            result["normals"] = self.normals.flatten().tolist()
        if self.vertex_colors is not None:
            result["colors"] = (self.vertex_colors / 255.0).flatten().tolist()
        if self.uvs is not None:
            result["uvs"] = self.uvs.flatten().tolist()
        return result


def load_ply(filepath: str) -> tuple:
    """
    Load PLY file (ASCII or binary).
    Returns (PointCloud, TriMesh or None).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    with open(filepath, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        is_binary_le = False
        is_binary_be = False
        n_vertices = 0
        n_faces = 0
        vertex_props = []
        face_props = []
        current_element = None

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'format':
                if 'binary_little_endian' in line:
                    is_binary_le = True
                elif 'binary_big_endian' in line:
                    is_binary_be = True
            elif parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    n_vertices = int(parts[2])
                elif current_element == 'face':
                    n_faces = int(parts[2])
            elif parts[0] == 'property':
                if current_element == 'vertex':
                    dtype = parts[1]
                    name = parts[2] if len(parts) > 2 else ''
                    vertex_props.append((name, dtype))
                elif current_element == 'face':
                    face_props.append(parts[1:])

        # Property name mapping
        prop_names = [p[0] for p in vertex_props]

        def get_prop_idx(names):
            for n in names:
                if n in prop_names:
                    return prop_names.index(n)
            return -1

        xi = get_prop_idx(['x', 'X'])
        yi = get_prop_idx(['y', 'Y'])
        zi = get_prop_idx(['z', 'Z'])
        ri = get_prop_idx(['red', 'r', 'diffuse_red'])
        gi = get_prop_idx(['green', 'g', 'diffuse_green'])
        bi = get_prop_idx(['blue', 'b', 'diffuse_blue'])
        nxi = get_prop_idx(['nx'])
        nyi = get_prop_idx(['ny'])
        nzi = get_prop_idx(['nz'])

        points = np.zeros((n_vertices, 3), dtype=np.float32)
        colors = np.zeros((n_vertices, 3), dtype=np.uint8) if ri >= 0 else None
        normals = np.zeros((n_vertices, 3), dtype=np.float32) if nxi >= 0 else None

        if is_binary_le or is_binary_be:
            endian_char = '<' if is_binary_le else '>'
            # Build numpy structured dtype for bulk read
            np_dtype_map = {
                'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
                'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
                'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
            }
            dt_fields = []
            for name, dtype in vertex_props:
                np_dt = np_dtype_map.get(dtype, 'f4')
                dt_fields.append((name, f'{endian_char}{np_dt}'))
            vertex_dtype = np.dtype(dt_fields)

            # Bulk read all vertices at once (numpy.frombuffer)
            raw_bytes = f.read(n_vertices * vertex_dtype.itemsize)
            raw = np.frombuffer(raw_bytes, dtype=vertex_dtype, count=n_vertices)

            if xi >= 0:
                x_name = prop_names[xi]
                y_name = prop_names[yi]
                z_name = prop_names[zi]
                points[:, 0] = raw[x_name].astype(np.float32)
                points[:, 1] = raw[y_name].astype(np.float32)
                points[:, 2] = raw[z_name].astype(np.float32)
            if colors is not None:
                r_name = prop_names[ri]
                g_name = prop_names[gi]
                b_name = prop_names[bi]
                colors[:, 0] = raw[r_name].astype(np.uint8)
                colors[:, 1] = raw[g_name].astype(np.uint8)
                colors[:, 2] = raw[b_name].astype(np.uint8)
            if normals is not None:
                nx_name = prop_names[nxi]
                ny_name = prop_names[nyi]
                nz_name = prop_names[nzi]
                normals[:, 0] = raw[nx_name].astype(np.float32)
                normals[:, 1] = raw[ny_name].astype(np.float32)
                normals[:, 2] = raw[nz_name].astype(np.float32)

            # Read faces
            faces_list = []
            for _ in range(n_faces):
                count = struct.unpack(f'{endian_char}B', f.read(1))[0]
                face = struct.unpack(f'{endian_char}{count}i', f.read(count * 4))
                if count == 3:
                    faces_list.append(face)
                elif count == 4:
                    faces_list.append((face[0], face[1], face[2]))
                    faces_list.append((face[0], face[2], face[3]))

        else:
            # ASCII
            for i in range(n_vertices):
                line = f.readline().decode('ascii').strip().split()
                vals = [float(v) for v in line]
                if xi >= 0:
                    points[i] = [vals[xi], vals[yi], vals[zi]]
                if colors is not None and ri < len(vals):
                    colors[i] = [int(vals[ri]), int(vals[gi]), int(vals[bi])]
                if normals is not None and nxi < len(vals):
                    normals[i] = [vals[nxi], vals[nyi], vals[nzi]]

            faces_list = []
            for _ in range(n_faces):
                line = f.readline().decode('ascii').strip().split()
                vals = [int(v) for v in line]
                count = vals[0]
                if count == 3:
                    faces_list.append(vals[1:4])
                elif count == 4:
                    faces_list.append([vals[1], vals[2], vals[3]])
                    faces_list.append([vals[1], vals[3], vals[4]])

    pc = PointCloud(
        points=points,
        colors=colors,
        normals=normals,
    )
    pc.compute_bounds()

    mesh = None
    if faces_list:
        faces = np.array(faces_list, dtype=np.int32)
        mesh = TriMesh(
            vertices=points.copy(),
            faces=faces,
            normals=normals.copy() if normals is not None else None,
            vertex_colors=colors.copy() if colors is not None else None,
        )
        mesh.compute_bounds()
        if mesh.normals is None:
            mesh.compute_normals()

    return pc, mesh


def load_obj(filepath: str) -> TriMesh:
    """Load OBJ mesh file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices = []
    normals = []
    uvs = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn' and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face_verts.append(idx)
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])

    mesh = TriMesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float32) if normals else None,
        uvs=np.array(uvs, dtype=np.float32) if uvs else None,
    )
    mesh.compute_bounds()
    if mesh.normals is None or len(mesh.normals) != len(mesh.vertices):
        mesh.compute_normals()
    return mesh


def load_las(filepath: str) -> PointCloud:
    """Load LAS/LAZ LiDAR point cloud (basic parser for LAS 1.2-1.4)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"LAS file not found: {filepath}")

    with open(filepath, 'rb') as f:
        # Public header
        signature = f.read(4)
        if signature != b'LASF':
            raise ValueError("Not a valid LAS file")

        f.read(2)  # file source id
        f.read(2)  # global encoding
        f.read(16)  # project id
        f.read(2)  # version major/minor
        version_major = struct.unpack('B', f.read(1))[0]
        version_minor = struct.unpack('B', f.read(1))[0]
        f.seek(94)  # skip to offset to point data
        # Re-read from proper position
        f.seek(96)
        offset_to_points = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # num variable length records
        point_format = struct.unpack('<B', f.read(1))[0]
        point_record_len = struct.unpack('<H', f.read(2))[0]
        num_points = struct.unpack('<I', f.read(4))[0]

        # Scale and offset
        f.seek(131)
        x_scale, y_scale, z_scale = struct.unpack('<3d', f.read(24))
        x_offset, y_offset, z_offset = struct.unpack('<3d', f.read(24))

        # Read points
        f.seek(offset_to_points)
        points = np.zeros((num_points, 3), dtype=np.float64)
        colors = None
        intensity = np.zeros(num_points, dtype=np.float32)

        has_color = point_format in (2, 3, 5, 7, 8, 10)
        if has_color:
            colors = np.zeros((num_points, 3), dtype=np.uint8)

        for i in range(num_points):
            record_start = f.tell()
            xi, yi, zi = struct.unpack('<3i', f.read(12))
            points[i] = [
                xi * x_scale + x_offset,
                yi * y_scale + y_offset,
                zi * z_scale + z_offset,
            ]
            intens = struct.unpack('<H', f.read(2))[0]
            intensity[i] = intens / 65535.0

            if has_color:
                # Color offset depends on format
                if point_format == 2:
                    f.seek(record_start + 20)
                elif point_format == 3:
                    f.seek(record_start + 28)
                else:
                    f.seek(record_start + 28)
                r, g, b = struct.unpack('<3H', f.read(6))
                colors[i] = [r >> 8, g >> 8, b >> 8]

            f.seek(record_start + point_record_len)

    pc = PointCloud(
        points=points.astype(np.float32),
        colors=colors,
        intensity=intensity,
    )
    pc.compute_bounds()
    return pc
