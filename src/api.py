import asyncio
import io
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from PIL import Image
from pydantic import BaseModel, Field

from src.predict import load_model, predict_image

MAX_FRAME_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB


def process_image_bytes(data: bytes) -> tuple[Image.Image, int, int]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img, img.width, img.height


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
        print("🟢 Model loaded successfully into app.state.")
    except Exception as e:
        print(f"🔴 Failed to load model artifact: {e}")
        app.state.model = None
    yield
    app.state.model = None
    print("🧹 App state cleared on shutdown.")


app = FastAPI(
    title="Produce Freshness Scanner API",
    description="Dual-mode produce freshness classifier — on-demand and live stream.",
    version="0.2.0",
    lifespan=lifespan,
)

@app.get("/ping")
async def ping():
    return {"status": "healthy"}


class SingleShotResponse(BaseModel):
    filename: str = Field(..., examples=["apple.jpg"])
    content_type: str = Field(..., examples=["image/jpeg"])
    status: str = Field(..., examples=["FRESH"])
    confidence: float = Field(..., examples=[0.98])
    resolution: str = Field(..., examples=["1920x1080"])


@app.get("/")
def health_check():
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "healthy",
        "service": "Produce Freshness Scanner API",
        "model_loaded": model_loaded,
    }


@app.post("/predict-image", response_model=SingleShotResponse)
async def predict_single_shot(file: Annotated[UploadFile, File(...)]):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only JPEG and PNG images are supported.",
        )

    try:
        contents = await file.read()

        if len(contents) > MAX_FRAME_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds maximum allowable limit (2MB).",
            )

        pil_img, width, height = await anyio.to_thread.run_sync(
            process_image_bytes, contents
        )

        model = getattr(app.state, "model", None)
        result = await anyio.to_thread.run_sync(predict_image, model, pil_img)

        return SingleShotResponse(
            filename=file.filename or "capture.jpg",
            content_type=file.content_type or "image/jpeg",
            status=result["status"],
            confidence=result["confidence"],
            resolution=f"{width}x{height}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process image: {str(e)}",
        )


@app.websocket("/ws/predict-stream")
async def predict_stream(websocket: WebSocket):
    await websocket.accept()

    frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)

    async def socket_reader():
        try:
            while True:
                data = await websocket.receive_bytes()
                if len(data) > MAX_FRAME_SIZE_BYTES:
                    continue
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(data)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"⚠️ Socket reader error: {e}")
        finally:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await frame_queue.put(None)

    reader_task = asyncio.create_task(socket_reader())

    try:
        while True:
            data = await frame_queue.get()

            if data is None:
                print("🔌 Disconnect signal received. Closing consumer.")
                break

            try:
                pil_img, width, height = await anyio.to_thread.run_sync(
                    process_image_bytes, data
                )
                model = getattr(websocket.app.state, "model", None)
                result = await anyio.to_thread.run_sync(predict_image, model, pil_img)

                await websocket.send_json({
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "frame_width": width,
                    "frame_height": height,
                })
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                try:
                    await websocket.send_json({
                        "error": "CORRUPT_FRAME",
                        "detail": "Failed to decode binary frame bytes.",
                    })
                except (WebSocketDisconnect, RuntimeError):
                    break
    finally:
        reader_task.cancel()
        await asyncio.gather(reader_task, return_exceptions=True)
        print("🧹 WebSocket cleaned up.")