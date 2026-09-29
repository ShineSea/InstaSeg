from __future__ import annotations

import logging
from typing import Protocol

import grpc
import numpy as np

from .generated import instance_segmentation_pb2 as pb2
from .generated import instance_segmentation_pb2_grpc as pb2_grpc

LOGGER = logging.getLogger(__name__)


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, x: int, y: int) -> np.ndarray: ...


class InstanceSegmentationService(pb2_grpc.InstanceSegmentationServicer):
    def __init__(self, segmenter: Segmenter, max_image_pixels: int) -> None:
        self._segmenter = segmenter
        self._max_image_pixels = max_image_pixels

    def Segment(
        self, request: pb2.SegmentRequest, context: grpc.ServicerContext
    ) -> pb2.SegmentResponse:
        width, height = int(request.width), int(request.height)
        if width == 0 or height == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "width and height must be non-zero")
        pixels = width * height
        if pixels > self._max_image_pixels:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"image has {pixels} pixels; limit is {self._max_image_pixels}",
            )
        expected_bytes = pixels * 3
        if len(request.rgb8) != expected_bytes:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"rgb8 has {len(request.rgb8)} bytes; expected {expected_bytes}",
            )
        if request.x >= width or request.y >= height:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"point ({request.x}, {request.y}) is outside {width}x{height} image",
            )

        image = np.frombuffer(request.rgb8, dtype=np.uint8).reshape(height, width, 3)
        try:
            mask = self._segmenter.segment(image, int(request.x), int(request.y))
        except Exception as exc:
            LOGGER.exception("SAM 2 inference failed")
            context.abort(grpc.StatusCode.INTERNAL, f"segmentation failed: {exc}")

        mask = np.asarray(mask)
        if mask.shape != (height, width):
            LOGGER.error("Invalid model mask shape: %s", mask.shape)
            context.abort(grpc.StatusCode.INTERNAL, "model returned an invalid mask shape")
        return pb2.SegmentResponse(mask=np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
