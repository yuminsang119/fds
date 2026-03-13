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

app = FastAPI(title="FDS Web Viewer", version="1.0.0")

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


# Mount static files last
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
