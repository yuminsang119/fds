"""
FDS Web Viewer - Backend Server
FastAPI server providing REST API and WebSocket for FDS simulation visualization.
"""

import json
import asyncio
import tempfile
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fds_parser import parse_smv_file, generate_demo_data

# GPU modules (optional - graceful fallback)
try:
    from gpu.volume_renderer import MultiGPUVolumeRenderer, GPUConfig
    from gpu.data_pipeline import GPUDataPipeline
    from gpu.post_processor import GPUPostProcessor
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Spatial modules
from spatial.scan_loader import load_ply, load_obj, load_las, PointCloud, TriMesh
from spatial.alignment import (
    SpatialTransform, compute_auto_alignment,
    compute_marker_alignment, compute_icp,
)

app = FastAPI(title="FDS Web Viewer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Cache for demo data
_demo_cache = None

# Cache for scan data
_scan_cache = {"point_cloud": None, "mesh": None, "transform": None}


def get_demo_data():
    global _demo_cache
    if _demo_cache is None:
        _demo_cache = generate_demo_data()
    return _demo_cache


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/demo")
async def demo_metadata():
    """Return demo simulation metadata."""
    data = get_demo_data()
    return JSONResponse({
        "title": data["title"],
        "chid": data["chid"],
        "bounds": data["bounds"],
        "mesh": data["mesh"],
        "n_slice_frames": len(data["slices"]),
        "n_volume_frames": len(data["volume_frames"]),
        "quantities": ["TEMPERATURE"],
    })


@app.get("/api/demo/slice/{frame_idx}")
async def demo_slice_frame(frame_idx: int):
    """Return a single slice frame."""
    data = get_demo_data()
    if 0 <= frame_idx < len(data["slices"]):
        return JSONResponse(data["slices"][frame_idx])
    return JSONResponse({"error": "Frame not found"}, status_code=404)


@app.get("/api/demo/volume/{frame_idx}")
async def demo_volume_frame(frame_idx: int):
    """Return a single volume frame."""
    data = get_demo_data()
    if 0 <= frame_idx < len(data["volume_frames"]):
        return JSONResponse(data["volume_frames"][frame_idx])
    return JSONResponse({"error": "Frame not found"}, status_code=404)


@app.get("/api/smv/{smv_file:path}")
async def parse_smv(smv_file: str):
    """Parse and return SMV file metadata."""
    path = Path(smv_file)
    if not path.exists() or not path.suffix == ".smv":
        return JSONResponse({"error": "SMV file not found"}, status_code=404)
    result = parse_smv_file(str(path))
    # Convert dataclasses for JSON
    meshes = []
    for m in result["meshes"]:
        meshes.append({
            "index": m.index,
            "i_range": m.i_range,
            "j_range": m.j_range,
            "k_range": m.k_range,
        })
    return JSONResponse({
        "title": result["title"],
        "chid": result["chid"],
        "bounds": result["bounds"],
        "n_meshes": len(meshes),
        "meshes": meshes,
        "n_slices": len(result["slices"]),
        "n_smoke3d": len(result["smoke3d"]),
    })


# ── 3D Scan API ──────────────────────────────────────────────────────

@app.post("/api/scan/upload")
async def upload_scan(file: UploadFile = File(...)):
    """Upload a 3D scan file (PLY, OBJ, LAS)."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ('.ply', '.obj', '.las', '.laz'):
        return JSONResponse(
            {"error": f"Unsupported format: {suffix}. Use PLY, OBJ, or LAS."},
            status_code=400,
        )

    # Save uploaded file
    dest = UPLOAD_DIR / file.filename
    with open(dest, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        if suffix == '.ply':
            pc, mesh = load_ply(str(dest))
            _scan_cache["point_cloud"] = pc
            _scan_cache["mesh"] = mesh
            result = {
                "format": "ply",
                "num_points": len(pc.points),
                "has_colors": pc.colors is not None,
                "has_normals": pc.normals is not None,
                "has_mesh": mesh is not None,
                "num_faces": len(mesh.faces) if mesh else 0,
                "bounds_min": pc.bounds_min.tolist(),
                "bounds_max": pc.bounds_max.tolist(),
            }
        elif suffix == '.obj':
            mesh = load_obj(str(dest))
            _scan_cache["point_cloud"] = PointCloud(
                points=mesh.vertices,
                colors=mesh.vertex_colors,
                normals=mesh.normals,
            )
            _scan_cache["point_cloud"].compute_bounds()
            _scan_cache["mesh"] = mesh
            result = {
                "format": "obj",
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "has_colors": mesh.vertex_colors is not None,
                "has_uvs": mesh.uvs is not None,
                "bounds_min": mesh.bounds_min.tolist(),
                "bounds_max": mesh.bounds_max.tolist(),
            }
        elif suffix in ('.las', '.laz'):
            pc = load_las(str(dest))
            _scan_cache["point_cloud"] = pc
            _scan_cache["mesh"] = None
            result = {
                "format": "las",
                "num_points": len(pc.points),
                "has_colors": pc.colors is not None,
                "has_intensity": pc.intensity is not None,
                "bounds_min": pc.bounds_min.tolist(),
                "bounds_max": pc.bounds_max.tolist(),
            }
        else:
            return JSONResponse({"error": "Unsupported format"}, status_code=400)

        return JSONResponse({"status": "ok", "filename": file.filename, **result})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/scan/points")
async def get_scan_points(max_points: int = 500000):
    """Get scan point cloud data for rendering."""
    pc = _scan_cache.get("point_cloud")
    if pc is None:
        return JSONResponse({"error": "No scan loaded"}, status_code=404)
    return JSONResponse(pc.to_json(max_points=max_points))


@app.get("/api/scan/mesh")
async def get_scan_mesh(max_faces: int = 200000):
    """Get scan mesh data for rendering."""
    mesh = _scan_cache.get("mesh")
    if mesh is None:
        return JSONResponse({"error": "No mesh available"}, status_code=404)
    return JSONResponse(mesh.to_json(max_faces=max_faces))


@app.post("/api/scan/align/auto")
async def auto_align_scan():
    """Auto-align scan bounding box to FDS domain."""
    pc = _scan_cache.get("point_cloud")
    if pc is None:
        return JSONResponse({"error": "No scan loaded"}, status_code=404)

    data = get_demo_data()
    b = data["bounds"]
    fds_min = np.array([b["x"][0], b["y"][0], b["z"][0]])
    fds_max = np.array([b["x"][1], b["y"][1], b["z"][1]])

    transform = compute_auto_alignment(
        pc.bounds_min, pc.bounds_max, fds_min, fds_max
    )
    _scan_cache["transform"] = transform
    return JSONResponse(transform.to_dict())


@app.post("/api/scan/align/markers")
async def marker_align_scan(data: dict):
    """Align using corresponding marker pairs."""
    scan_markers = np.array(data["scan_markers"], dtype=np.float64)
    fds_markers = np.array(data["fds_markers"], dtype=np.float64)

    if len(scan_markers) < 3:
        return JSONResponse({"error": "Need at least 3 marker pairs"}, status_code=400)

    transform = compute_marker_alignment(scan_markers, fds_markers)
    _scan_cache["transform"] = transform
    return JSONResponse(transform.to_dict())


@app.post("/api/scan/align/icp")
async def icp_align_scan():
    """Refine alignment using ICP registration."""
    pc = _scan_cache.get("point_cloud")
    if pc is None:
        return JSONResponse({"error": "No scan loaded"}, status_code=404)

    # Use FDS domain boundary points as target
    data = get_demo_data()
    b = data["bounds"]
    # Generate target points from FDS domain
    nx, ny, nz = 20, 20, 10
    x = np.linspace(b["x"][0], b["x"][1], nx)
    y = np.linspace(b["y"][0], b["y"][1], ny)
    z = np.linspace(b["z"][0], b["z"][1], nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    target_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Apply existing transform first if any
    source = pc.points.copy()
    existing = _scan_cache.get("transform")
    if existing:
        source = existing.apply(source)

    transform = compute_icp(source, target_points, max_iterations=30)
    _scan_cache["transform"] = transform
    return JSONResponse(transform.to_dict())


@app.post("/api/scan/transform")
async def set_scan_transform(data: dict):
    """Set manual transform for the scan."""
    transform = SpatialTransform.from_dict(data)
    _scan_cache["transform"] = transform
    return JSONResponse(transform.to_dict())


@app.get("/api/scan/transform")
async def get_scan_transform():
    """Get current scan transform."""
    transform = _scan_cache.get("transform")
    if transform is None:
        transform = SpatialTransform()
    return JSONResponse(transform.to_dict())


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Stream simulation frames via WebSocket (JSON mode for compatibility)."""
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)

            if cmd.get("action") == "stream_slices":
                data = get_demo_data()
                speed = cmd.get("speed", 1.0)
                for i, frame in enumerate(data["slices"]):
                    await websocket.send_json({
                        "type": "slice_frame",
                        "frame_index": i,
                        "total_frames": len(data["slices"]),
                        **frame,
                    })
                    await asyncio.sleep(0.1 / speed)
                await websocket.send_json({"type": "stream_complete"})

            elif cmd.get("action") == "stream_volume":
                data = get_demo_data()
                speed = cmd.get("speed", 1.0)
                for i, frame in enumerate(data["volume_frames"]):
                    await websocket.send_json({
                        "type": "volume_frame",
                        "frame_index": i,
                        "total_frames": len(data["volume_frames"]),
                        **frame,
                    })
                    await asyncio.sleep(0.2 / speed)
                await websocket.send_json({"type": "stream_complete"})

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/binary_stream")
async def binary_stream(websocket: WebSocket):
    """
    Binary WebSocket streaming with backpressure.
    Protocol:
      Client sends JSON: {"action": "stream_slices|stream_volume", "speed": 1.0}
      Server sends binary: [4-byte header_len][JSON header][float32 data]
      Client sends "ACK" after each frame (backpressure).
    """
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            data = get_demo_data()

            if cmd.get("action") == "stream_slices":
                frames = data["slices"]
                frame_type = "slice_frame"
            elif cmd.get("action") == "stream_volume":
                frames = data["volume_frames"]
                frame_type = "volume_frame"
            else:
                continue

            for i, frame in enumerate(frames):
                # Header (small JSON)
                header = json.dumps({
                    "type": frame_type,
                    "frame_index": i,
                    "total_frames": len(frames),
                    "time": frame.get("time", 0),
                    "nx": frame.get("nx", 0),
                    "ny": frame.get("ny", 0),
                    "nz": frame.get("nz", 0),
                    "min_val": frame.get("min_val", 0),
                    "max_val": frame.get("max_val", 0),
                }).encode('utf-8')

                # Binary payload: [header_len(4B)][header][float32 data]
                frame_data = np.array(frame["data"], dtype=np.float32).tobytes()
                import struct as _struct
                payload = _struct.pack('<I', len(header)) + header + frame_data
                await websocket.send_bytes(payload)

                # Wait for client ACK (backpressure)
                try:
                    ack = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                except asyncio.TimeoutError:
                    break  # Client not responding

            # Signal completion
            end_header = json.dumps({"type": "stream_complete"}).encode('utf-8')
            import struct as _struct
            await websocket.send_bytes(_struct.pack('<I', len(end_header)) + end_header)

    except WebSocketDisconnect:
        pass


# ── GPU Rendering API ────────────────────────────────────────────────

_gpu_renderer = None
_gpu_pipeline = None
_gpu_postproc = None


def get_gpu_renderer():
    global _gpu_renderer
    if _gpu_renderer is None and HAS_GPU:
        _gpu_renderer = MultiGPUVolumeRenderer(GPUConfig(
            num_gpus=4,
            render_resolution=(1920, 1080),
            volume_resolution=(512, 512, 256),
            ray_samples=256,
            use_fp16=True,
        ))
    return _gpu_renderer


def get_gpu_pipeline():
    global _gpu_pipeline
    if _gpu_pipeline is None and HAS_GPU:
        _gpu_pipeline = GPUDataPipeline()
    return _gpu_pipeline


def get_gpu_postproc():
    global _gpu_postproc
    if _gpu_postproc is None and HAS_GPU:
        _gpu_postproc = GPUPostProcessor(gpu_index=2)
    return _gpu_postproc


@app.get("/api/gpu/status")
async def gpu_status():
    """Return GPU availability and status."""
    if not HAS_GPU:
        return JSONResponse({"available": False, "reason": "GPU modules not installed"})

    renderer = get_gpu_renderer()
    pipeline = get_gpu_pipeline()
    return JSONResponse({
        "available": True,
        "renderer": renderer.get_status() if renderer else {},
        "pipeline": pipeline.get_memory_stats() if pipeline else {},
    })


@app.post("/api/gpu/render")
async def gpu_render_frame(params: dict = None):
    """Server-side GPU render a single frame."""
    import base64
    import io

    if not HAS_GPU:
        return JSONResponse({"error": "GPU not available"}, status_code=503)

    renderer = get_gpu_renderer()
    if not renderer or not renderer.initialized:
        return JSONResponse({"error": "GPU renderer not initialized"}, status_code=503)

    camera = params or {
        "position": [5, 5, 5],
        "target": [2.5, 1.5, 2.5],
        "fov": 50,
        "up": [0, 1, 0],
    }

    frame = renderer.render_frame(camera)

    # Apply post-processing
    postproc = get_gpu_postproc()
    if postproc:
        frame = postproc.process(frame, {
            "tone_mapping": True,
            "bloom": True,
            "exposure": 1.2,
            "gamma": 2.2,
        })

    # Encode as JPEG for streaming
    try:
        from PIL import Image
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({
            "frame": encoded,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "format": "jpeg",
        })
    except ImportError:
        # Fallback: raw data
        return JSONResponse({
            "frame": frame.tolist(),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "format": "raw",
        })


@app.post("/api/gpu/upload_volume")
async def gpu_upload_volume():
    """Upload demo volume data to GPU for rendering."""
    if not HAS_GPU:
        return JSONResponse({"error": "GPU not available"}, status_code=503)

    import numpy as np

    renderer = get_gpu_renderer()
    pipeline = get_gpu_pipeline()
    data = get_demo_data()

    if data["volume_frames"]:
        frame = data["volume_frames"][0]
        vol = np.array(frame["data"]).reshape(frame["nz"], frame["ny"], frame["nx"])

        # Preprocess on GPU 3
        if pipeline:
            processed = pipeline.preprocess_volume(vol, target_shape=(256, 256, 128))
            vol = processed["volume"]

        # Upload to render GPUs
        result = renderer.upload_volume(vol)
        return JSONResponse(result)

    return JSONResponse({"error": "No volume data"}, status_code=404)


@app.websocket("/ws/gpu_stream")
async def gpu_websocket_stream(websocket: WebSocket):
    """Stream GPU-rendered frames via WebSocket (real-time)."""
    import base64
    import io

    await websocket.accept()

    if not HAS_GPU:
        await websocket.send_json({"error": "GPU not available"})
        await websocket.close()
        return

    renderer = get_gpu_renderer()
    postproc = get_gpu_postproc()

    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)

            if cmd.get("action") == "render":
                camera = cmd.get("camera", {
                    "position": [5, 5, 5],
                    "target": [2.5, 1.5, 2.5],
                    "fov": 50,
                    "up": [0, 1, 0],
                })

                frame = renderer.render_frame(camera)
                if postproc:
                    frame = postproc.process(frame, cmd.get("effects", {}))

                try:
                    from PIL import Image
                    img = Image.fromarray(frame)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    encoded = base64.b64encode(buf.getvalue()).decode()
                    await websocket.send_json({
                        "type": "gpu_frame",
                        "frame": encoded,
                        "format": "jpeg",
                    })
                except ImportError:
                    await websocket.send_json({
                        "type": "gpu_frame",
                        "format": "unavailable",
                    })

    except WebSocketDisconnect:
        pass


# Mount static files last
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
