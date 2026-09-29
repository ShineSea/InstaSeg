from __future__ import annotations

import logging
import signal
from concurrent import futures

import grpc

from .config import Settings
from .generated import instance_segmentation_pb2_grpc as pb2_grpc
from .predictor import Sam21Segmenter
from .service import InstanceSegmentationService

LOGGER = logging.getLogger(__name__)


def create_server(settings: Settings, segmenter: Sam21Segmenter) -> grpc.Server:
    # RGB payload is 3 bytes/pixel; leave a small allowance for protobuf fields.
    max_message_bytes = settings.max_image_pixels * 3 + 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.max_workers),
        options=[
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", settings.max_image_pixels + 1024),
        ],
    )
    pb2_grpc.add_InstanceSegmentationServicer_to_server(
        InstanceSegmentationService(segmenter, settings.max_image_pixels), server
    )
    return server


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Loading SAM 2.1 model %s on %s", settings.model_id, settings.device)
    segmenter = Sam21Segmenter.load(settings)
    server = create_server(settings, segmenter)
    address = f"{settings.host}:{settings.port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"could not bind gRPC server to {address}")

    def stop(_signum: int, _frame: object) -> None:
        LOGGER.info("Stopping gRPC server")
        server.stop(grace=10)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.start()
    LOGGER.info("InstanceSegmentation listening on %s", address)
    server.wait_for_termination()


if __name__ == "__main__":
    main()

