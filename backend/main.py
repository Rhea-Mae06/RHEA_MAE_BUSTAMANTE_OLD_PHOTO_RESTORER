from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    LAMA_MODEL_PATH,
    MAX_UPLOAD_BYTES,
    PROJECT_DIR,
)
from face_detection import get_face_detector
from inpainting import get_inpainter
from restoration import RestorationError, restore_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("old-photo-restorer")

app = FastAPI(title="Old Photo Restoration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Restore-Engine",
        "X-Restore-Damage-Percent",
        "X-Restore-Damage-Pixels",
        "X-Restore-Faces",
        "X-Restore-Width",
        "X-Restore-Height",
        "X-Restore-Ms",
        "X-Restore-Grayscale",
        "X-Restore-Mode",
        "X-Restore-Tiles",
        "X-Restore-Protected-Face-Pixels",
    ],
    max_age=86400,
)


@app.middleware("http")
async def protect_private_paths(request, call_next):
    path = request.url.path
    if (
        path.startswith("/backend")
        or path.startswith("/.git")
        or "/.venv" in path
        or path.endswith(".py")
    ):
        return JSONResponse(status_code=404, content={"error": "Not found."})
    return await call_next(request)


@app.get("/health")
async def health():
    inpainter = get_inpainter()
    return {
        "status": "ok",
        "lama_model_present": inpainter.model_present,
        "lama_model_loaded": inpainter.loaded,
        "lama_model_error": inpainter.load_error,
        "face_detector_available": get_face_detector().available(),
        "ai_inpainting": True,
    }


@app.post("/restore")
async def restore(
    file: UploadFile = File(...),
    sensitivity: float = Form(0.5),
):
    if sensitivity < 0.0 or sensitivity > 1.0:
        return JSONResponse(
            status_code=400,
            content={"error": "Sensitivity must be between 0 and 1."},
        )
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        return JSONResponse(
            status_code=415,
            content={"error": "Please upload an image file (JPG, PNG, WebP, BMP or TIFF)."},
        )

    try:
        data = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=400, content={"error": f"Could not read the uploaded file: {exc}"}
        )
    if len(data) == 0:
        return JSONResponse(status_code=400, content={"error": "The uploaded file is empty."})
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "The uploaded image is too large. The maximum size is 60 MB."
            },
        )

    try:
        result = await run_in_threadpool(restore_image, data, sensitivity)
    except RestorationError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except MemoryError:
        return JSONResponse(
            status_code=413,
            content={"error": "There was not enough memory to restore this image."},
        )
    except Exception as exc:
        logger.exception("Restoration failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Restoration failed: {exc}"},
        )

    meta = result.meta
    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": 'inline; filename="ai-restored.png"',
        "X-Restore-Engine": str(meta["engine"]),
        "X-Restore-Damage-Percent": f"{meta['damage_ratio'] * 100:.4f}",
        "X-Restore-Damage-Pixels": str(meta["damage_pixels"]),
        "X-Restore-Faces": str(meta["faces"]),
        "X-Restore-Width": str(meta["width"]),
        "X-Restore-Height": str(meta["height"]),
        "X-Restore-Ms": str(meta["processing_ms"]),
        "X-Restore-Grayscale": "true" if meta["grayscale"] else "false",
        "X-Restore-Mode": str(meta["mode"]),
        "X-Restore-Tiles": str(meta["tiles"]),
        "X-Restore-Protected-Face-Pixels": str(meta["protected_face_pixels"]),
    }
    return Response(content=result.image_bytes, media_type="image/png", headers=headers)


@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    await file.close()
    return JSONResponse(
        status_code=503,
        content={
            "error": (
                "AI colourisation is not enabled in this build. "
                "Automatic restoration preserves the original colour or black-and-white appearance."
            )
        },
    )


app.mount(
    "/",
    StaticFiles(directory=str(PROJECT_DIR), html=True),
    name="frontend",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
