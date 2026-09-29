"""Regenerate the checked-in Python gRPC bindings from the project proto."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "instaseg" / "generated"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "instance_segmentation.proto", OUTPUT / "instance_segmentation.proto")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{OUTPUT}",
            f"--python_out={OUTPUT}",
            f"--grpc_python_out={OUTPUT}",
            str(OUTPUT / "instance_segmentation.proto"),
        ],
        check=True,
    )

    grpc_file = OUTPUT / "instance_segmentation_pb2_grpc.py"
    source = grpc_file.read_text(encoding="utf-8")
    source = re.sub(
        r"^import instance_segmentation_pb2 as instance__segmentation__pb2$",
        "from . import instance_segmentation_pb2 as instance__segmentation__pb2",
        source,
        flags=re.MULTILINE,
    )
    grpc_file.write_text(source, encoding="utf-8")
    (OUTPUT / "__init__.py").touch()


if __name__ == "__main__":
    main()

