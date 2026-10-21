from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

import numpy as np

from .config import Settings


class Sam21Segmenter:
    """Thread-safe adapter around SAM2ImagePredictor.

    SAM2ImagePredictor stores the current image embedding internally, so one
    predictor cannot safely execute concurrent set_image/predict sequences.
    """

    def __init__(self, predictor: Any, device: str) -> None:
        self._predictor = predictor
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings) -> Sam21Segmenter:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if settings.checkpoint and settings.model_config:
            model = build_sam2(
                settings.model_config,
                settings.checkpoint,
                device=device,
            )
            predictor = SAM2ImagePredictor(model)
        else:
            predictor = SAM2ImagePredictor.from_pretrained(
                settings.model_id,
                device=device,
            )
        return cls(predictor, device)

    def segment(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        return self._segment(
            image,
            point_coords=np.asarray([[x, y]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
        )

    def segment_box(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray:
        return self._segment(image, box=np.asarray(box, dtype=np.float32))

    def _segment(self, image: np.ndarray, **prompt: Any) -> np.ndarray:
        import torch

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.startswith("cuda")
            else nullcontext()
        )
        with self._lock, torch.inference_mode(), autocast:
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                **prompt,
                multimask_output=True,
            )

        best = int(np.argmax(np.asarray(scores)))
        mask = np.asarray(masks[best], dtype=np.bool_)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(
                f"SAM 2 returned mask shape {mask.shape}, expected {image.shape[:2]}"
            )
        return mask

