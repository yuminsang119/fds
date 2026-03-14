"""
FDS Web Viewer - Backend Server
FastAPI server providing REST API and WebSocket for FDS simulation visualization.
"""

import json
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

app = FastAPI(title="FDS Web Viewer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Cache for demo data
_demo_cache = None


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


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Stream simulation frames via WebSocket."""
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
