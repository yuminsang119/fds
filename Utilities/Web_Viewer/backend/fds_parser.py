"""
FDS Output File Parser
Parses .smv, .sf, .q, .s3d, and slice files from FDS simulations.
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MeshInfo:
    index: int
    i_range: tuple
    j_range: tuple
    k_range: tuple
    x_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    y_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    z_coords: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SliceInfo:
    mesh_index: int
    quantity: str
    short_name: str
    units: str
    i_range: tuple
    j_range: tuple
    k_range: tuple
    filename: str


@dataclass
class Smoke3DInfo:
    mesh_index: int
    quantity: str
    filename: str


def parse_smv_file(smv_path: str) -> dict:
    """Parse a Smokeview (.smv) file to extract simulation metadata."""
    smv_path = Path(smv_path)
    result = {
        "title": "",
        "chid": smv_path.stem,
        "meshes": [],
        "slices": [],
        "smoke3d": [],
        "bounds": {"x": [0, 1], "y": [0, 1], "z": [0, 1]},
    }

    if not smv_path.exists():
        return result

    lines = smv_path.read_text().splitlines()
    i = 0
    mesh_idx = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "TITLE":
            i += 1
            if i < len(lines):
                result["title"] = lines[i].strip()

        elif line.startswith("GRID"):
            i += 1
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    ni, nj, nk = int(parts[0]), int(parts[1]), int(parts[2])
                    mesh = MeshInfo(
                        index=mesh_idx,
                        i_range=(0, ni),
                        j_range=(0, nj),
                        k_range=(0, nk),
                    )
                    result["meshes"].append(mesh)
                    mesh_idx += 1

        elif line.startswith("TRNX") or line.startswith("TRNY") or line.startswith("TRNZ"):
            axis = line[3]
            i += 1
            if i < len(lines):
                n = int(lines[i].strip())
                coords = []
                for _ in range(n + 1):
                    i += 1
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 2:
                            coords.append(float(parts[1]))
                if result["meshes"] and coords:
                    mesh = result["meshes"][-1]
                    arr = np.array(coords)
                    if axis == "X":
                        mesh.x_coords = arr
                    elif axis == "Y":
                        mesh.y_coords = arr
                    elif axis == "Z":
                        mesh.z_coords = arr

        elif line.startswith("SLC"):
            # SLCF or SLCC
            i += 1
            if i < len(lines):
                parts = lines[i].strip().split()
                mesh_id = int(parts[0]) - 1 if parts else 0
            i += 1
            filename = lines[i].strip() if i < len(lines) else ""
            i += 1
            quantity = lines[i].strip() if i < len(lines) else ""
            i += 1
            short_name = lines[i].strip() if i < len(lines) else ""
            i += 1
            units = lines[i].strip() if i < len(lines) else ""

            slc = SliceInfo(
                mesh_index=mesh_id,
                quantity=quantity,
                short_name=short_name,
                units=units,
                i_range=(0, 0),
                j_range=(0, 0),
                k_range=(0, 0),
                filename=filename,
            )
            result["slices"].append(slc)

        elif line.startswith("SMOKE3D"):
            i += 1
            if i < len(lines):
                parts = lines[i].strip().split()
                mesh_id = int(parts[0]) - 1 if parts else 0
            i += 1
            filename = lines[i].strip() if i < len(lines) else ""
            i += 1
            quantity = lines[i].strip() if i < len(lines) else ""

            s3d = Smoke3DInfo(
                mesh_index=mesh_id,
                quantity=quantity,
                filename=filename,
            )
            result["smoke3d"].append(s3d)

        i += 1

    # Calculate bounds from meshes
    if result["meshes"]:
        all_x, all_y, all_z = [], [], []
        for m in result["meshes"]:
            if len(m.x_coords) > 0:
                all_x.extend([m.x_coords.min(), m.x_coords.max()])
            if len(m.y_coords) > 0:
                all_y.extend([m.y_coords.min(), m.y_coords.max()])
            if len(m.z_coords) > 0:
                all_z.extend([m.z_coords.min(), m.z_coords.max()])
        if all_x:
            result["bounds"]["x"] = [min(all_x), max(all_x)]
        if all_y:
            result["bounds"]["y"] = [min(all_y), max(all_y)]
        if all_z:
            result["bounds"]["z"] = [min(all_z), max(all_z)]

    return result


def read_slice_file(filepath: str, nx: int, ny: int) -> list:
    """Read a binary FDS slice file and return list of (time, data) tuples."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    frames = []
    with open(filepath, "rb") as f:
        while True:
            try:
                # Read Fortran record markers + time
                f.read(4)  # record marker
                time_val = struct.unpack("f", f.read(4))[0]
                f.read(4)  # record marker

                # Read data
                f.read(4)  # record marker
                n_vals = nx * ny
                data = np.frombuffer(f.read(n_vals * 4), dtype=np.float32)
                f.read(4)  # record marker

                frames.append({"time": float(time_val), "data": data.tolist()})
            except (struct.error, ValueError):
                break

    return frames


def generate_demo_data(nx=50, ny=50, nz=30, n_frames=60) -> dict:
    """Generate demo fire simulation data for testing the web viewer."""
    result = {
        "title": "FDS Demo - Room Fire Simulation",
        "chid": "room_fire_demo",
        "bounds": {"x": [0.0, 5.0], "y": [0.0, 5.0], "z": [0.0, 3.0]},
        "mesh": {"nx": nx, "ny": ny, "nz": nz},
        "slices": [],
        "volume_frames": [],
    }

    # Generate temperature slice data (XY plane at z=1.5m)
    for frame_idx in range(n_frames):
        t = frame_idx / 10.0
        temp_data = np.full((ny, nx), 20.0, dtype=np.float32)

        # Fire source at center
        cx, cy = nx // 2, ny // 2
        for iy in range(ny):
            for ix in range(nx):
                r = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                spread = min(t * 2, 10)
                if r < spread:
                    intensity = max(0, 1 - r / max(spread, 0.1))
                    temp_data[iy, ix] = 20 + 800 * intensity * min(t / 3.0, 1.0)
                    # Add turbulent fluctuation
                    temp_data[iy, ix] += np.random.normal(0, 20) * intensity

        result["slices"].append({
            "time": round(t, 2),
            "quantity": "TEMPERATURE",
            "units": "C",
            "plane": "z",
            "plane_value": 1.5,
            "nx": nx,
            "ny": ny,
            "data": temp_data.flatten().tolist(),
            "min_val": float(temp_data.min()),
            "max_val": float(temp_data.max()),
        })

    # Generate 3D smoke density volume data (fewer frames, lower res)
    vol_nx, vol_ny, vol_nz = nx // 2, ny // 2, nz // 2
    for frame_idx in range(0, n_frames, 3):
        t = frame_idx / 10.0
        smoke = np.zeros((vol_nz, vol_ny, vol_nx), dtype=np.float32)

        cx, cy = vol_nx // 2, vol_ny // 2
        for iz in range(vol_nz):
            z_frac = iz / vol_nz
            for iy in range(vol_ny):
                for ix in range(vol_nx):
                    r = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                    # Smoke rises and spreads
                    rise = min(t * 0.3, 1.0)
                    if z_frac < rise:
                        spread = 3 + z_frac * 5
                        if r < spread:
                            density = (1 - r / spread) * min(t / 2.0, 1.0)
                            density *= (0.5 + 0.5 * z_frac)  # denser at top
                            smoke[iz, iy, ix] = max(0, density + np.random.normal(0, 0.05))

        result["volume_frames"].append({
            "time": round(t, 2),
            "nx": vol_nx,
            "ny": vol_ny,
            "nz": vol_nz,
            "data": smoke.flatten().tolist(),
            "max_val": float(smoke.max()),
        })

    return result
