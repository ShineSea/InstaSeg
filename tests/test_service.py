from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from instaseg.generated import instance_segmentation_pb2 as pb2
from instaseg.generated import instance_segmentation_pb2_grpc as pb2_grpc
from instaseg.service import InstanceSegmentationService


class FakeSegmenter:
    def segment(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        mask[y, x] = True
        return mask


@pytest.fixture
def stub():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_InstanceSegmentationServicer_to_server(
        InstanceSegmentationService(FakeSegmenter(), max_image_pixels=100), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield pb2_grpc.InstanceSegmentationStub(channel)
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_segment_returns_flat_binary_mask(stub) -> None:
    response = stub.Segment(
        pb2.SegmentRequest(width=3, height=2, rgb8=bytes(3 * 2 * 3), x=1, y=1)
    )
    assert response.mask == bytes([0, 0, 0, 0, 1, 0])


@pytest.mark.parametrize(
    ("segment_request", "detail"),
    [
        (pb2.SegmentRequest(width=0, height=2), "non-zero"),
        (pb2.SegmentRequest(width=2, height=2, rgb8=b"bad"), "expected 12"),
        (pb2.SegmentRequest(width=2, height=2, rgb8=bytes(12), x=2), "outside"),
    ],
)
def test_invalid_requests_return_invalid_argument(stub, segment_request, detail: str) -> None:
    with pytest.raises(grpc.RpcError) as error:
        stub.Segment(segment_request)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert detail in error.value.details()
