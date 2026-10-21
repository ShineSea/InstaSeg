from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings, _positive_int
from .predictor import Sam21Segmenter

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("web")


class HttpSegmenter(Protocol):
    def segment(self, image: np.ndarray, x: int, y: int) -> np.ndarray: ...

    def segment_box(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class HttpSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_env(cls) -> HttpSettings:
        port = _positive_int("INSTASEG_HTTP_PORT", 8000)
        if port > 65535:
            raise ValueError("INSTASEG_HTTP_PORT must be at most 65535")
        return cls(
            host=os.getenv("INSTASEG_HTTP_HOST", "0.0.0.0"),
            port=port,
            max_upload_bytes=_positive_int("INSTASEG_HTTP_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
        )


def _decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=["JPEG", "PNG", "WEBP", "BMP"]) as source:
                if source.width * source.height > max_pixels:
                    raise HTTPException(413, f"图片像素数超过限制（{max_pixels:,}）")
                return np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "图片尺寸过大") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "无法读取图片，请上传有效的 JPEG、PNG、WebP 或 BMP 文件") from exc


def _parse_prompt(mode: str, raw: str, width: int, height: int) -> list[float]:
    try:
        values = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, "prompt 必须是 JSON 坐标数组") from exc
    expected = 2 if mode == "point" else 4
    if (
        not isinstance(values, list)
        or len(values) != expected
        or any(
            type(value) not in (int, float)
            or (isinstance(value, float) and not math.isfinite(value))
            for value in values
        )
    ):
        raise HTTPException(422, f"{mode} 模式需要 {expected} 个有限数字坐标")
    if mode == "point":
        x, y = values
        if not (0 <= x < width and 0 <= y < height):
            raise HTTPException(422, "提示点必须位于图片内")
    else:
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise HTTPException(422, "选框必须位于图片内，且满足 x1 < x2、y1 < y2")
    return values


def _png_url(pixels: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def create_app(
    settings: Settings | None = None,
    segmenter: HttpSegmenter | None = None,
    http_settings: HttpSettings | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    http_settings = http_settings or HttpSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        LOGGER.info("Loading HTTP segmentation model")
        app.state.segmenter = segmenter if segmenter is not None else Sam21Segmenter.load(settings)
        yield
        app.state.segmenter = None

    app = FastAPI(title="InstaSeg HTTP API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "max_image_pixels": settings.max_image_pixels,
            "max_upload_bytes": http_settings.max_upload_bytes,
        }

    @app.post("/api/segment")
    def segment(
        image: Annotated[UploadFile, File(description="JPEG / PNG / WebP / BMP")],
        mode: Annotated[Literal["point", "box"], Form()],
        prompt: Annotated[str, Form(description="[x,y] or [x1,y1,x2,y2] in original pixels")],
    ) -> dict:
        started = time.perf_counter()
        data = image.file.read(http_settings.max_upload_bytes + 1)
        if len(data) > http_settings.max_upload_bytes:
            raise HTTPException(413, "上传文件超过大小限制")
        rgb = _decode_image(data, settings.max_image_pixels)
        height, width = rgb.shape[:2]
        coords = _parse_prompt(mode, prompt, width, height)
        try:
            if mode == "point":
                mask = app.state.segmenter.segment(rgb, int(coords[0]), int(coords[1]))
            else:
                mask = app.state.segmenter.segment_box(rgb, tuple(coords))
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != (height, width):
                raise RuntimeError(f"Invalid model mask shape: {mask.shape}")
        except Exception as exc:
            LOGGER.exception("HTTP segmentation failed")
            raise HTTPException(500, "模型推理失败，请查看服务日志后重试") from exc

        overlay = rgb.copy()
        overlay[mask] = (rgb[mask] * 0.5 + np.array([34, 211, 153]) * 0.5).astype(np.uint8)
        alpha = mask.astype(np.uint8) * 255
        return {
            "width": width,
            "height": height,
            "foreground_pixels": int(mask.sum()),
            "mask": _png_url(alpha),
            "overlay": _png_url(overlay),
            "cutout": _png_url(np.dstack((rgb, alpha))),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return app


def main() -> None:
    settings = Settings.from_env()
    http_settings = HttpSettings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(settings, http_settings=http_settings),
        host=http_settings.host,
        port=http_settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
