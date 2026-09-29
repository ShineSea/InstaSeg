from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    model_id: str
    device: str
    checkpoint: str | None
    model_config: str | None
    max_workers: int
    max_image_pixels: int
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        checkpoint = os.getenv("INSTASEG_CHECKPOINT") or None
        model_config = os.getenv("INSTASEG_MODEL_CONFIG") or None
        if bool(checkpoint) != bool(model_config):
            raise ValueError(
                "INSTASEG_CHECKPOINT and INSTASEG_MODEL_CONFIG must be set together"
            )
        port = _positive_int("INSTASEG_PORT", 50051)
        if port > 65535:
            raise ValueError("INSTASEG_PORT must be at most 65535")
        return cls(
            host=os.getenv("INSTASEG_HOST", "0.0.0.0"),
            port=port,
            model_id=os.getenv("INSTASEG_MODEL_ID", "facebook/sam2.1-hiera-small"),
            device=os.getenv("INSTASEG_DEVICE", "auto"),
            checkpoint=checkpoint,
            model_config=model_config,
            max_workers=_positive_int("INSTASEG_MAX_WORKERS", 8),
            max_image_pixels=_positive_int("INSTASEG_MAX_IMAGE_PIXELS", 25_000_000),
            log_level=os.getenv("INSTASEG_LOG_LEVEL", "INFO").upper(),
        )

