from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from instaseg.config import Settings
from instaseg.http_server import HttpSettings, create_app


class FakeSegmenter:
    def __init__(self) -> None:
        self.calls = []
        self.fail = False
        self.bad_shape = False

    def segment(self, image, x, y):
        self.calls.append(("point", image.copy(), (x, y)))
        if self.fail:
            raise RuntimeError("private model error")
        if self.bad_shape:
            return np.zeros((1, 1))
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[y, x] = True
        return mask

    def segment_box(self, image, box):
        self.calls.append(("box", image.copy(), box))
        mask = np.zeros(image.shape[:2], dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = True
        return mask


@pytest.fixture
def client():
    fake = FakeSegmenter()
    app = create_app(Settings.from_env(), fake, HttpSettings(max_upload_bytes=1024))
    with TestClient(app) as session:
        yield session, fake


def image_file(size=(4, 3), *, orientation=None):
    output = io.BytesIO()
    image = Image.new("RGB", size, (100, 50, 20))
    if orientation is None:
        image.save(output, "PNG")
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, "JPEG", exif=exif)
    return ("sample.png", output.getvalue(), "image/png")


def decode_png(url):
    assert url.startswith("data:image/png;base64,")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))


def test_web_page_assets_and_health(client):
    session, _ = client
    assert "图像分割工作台" in session.get("/").text
    assert session.get("/static/app.js").status_code == 200
    assert session.get("/static/style.css").status_code == 200
    assert session.get("/api/health").json()["max_upload_bytes"] == 1024


@pytest.mark.parametrize(
    ("mode", "prompt", "pixels"),
    [
        ("point", "[2,1]", 1),
        ("box", "[1,1,4,3]", 6),
    ],
)
def test_segment_outputs_full_resolution_pngs(client, mode, prompt, pixels):
    session, fake = client
    response = session.post(
        "/api/segment", files={"image": image_file()}, data={"mode": mode, "prompt": prompt}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert (result["width"], result["height"]) == (4, 3)
    assert result["foreground_pixels"] == pixels
    assert result["elapsed_ms"] >= 0
    mask = decode_png(result["mask"])
    assert mask.shape == (3, 4)
    assert set(np.unique(mask)) == {0, 255}
    assert (mask > 0).sum() == pixels
    cutout = decode_png(result["cutout"])
    assert cutout.shape == (3, 4, 4)
    np.testing.assert_array_equal(cutout[:, :, 3], mask)
    overlay = decode_png(result["overlay"])
    np.testing.assert_array_equal(overlay[mask == 0], fake.calls[0][1][mask == 0])
    assert fake.calls[0][0] == mode
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("mode", "prompt"),
    [
        ("point", "[-1,0]"),
        ("point", "[4,0]"),
        ("point", "[0,3]"),
        ("point", "[true,0]"),
        ("point", "[NaN,0]"),
        ("point", "[Infinity,0]"),
        ("point", "[1e999,0]"),
        ("point", '["1",0]'),
        ("point", "[0]"),
        ("point", f"[{10**400},0]"),
        ("point", "{}"),
        ("point", "invalid"),
        ("point", "[0,0,1,1]"),
        ("box", "[2,1,1,2]"),
        ("box", "[0,0,0,1]"),
        ("box", "[0,0,5,3]"),
        ("box", "[0,-1,4,3]"),
        ("box", "[0,0]"),
        ("other", "[0,0]"),
    ],
)
def test_invalid_prompts_never_reach_model(client, mode, prompt):
    session, fake = client
    response = session.post(
        "/api/segment", files={"image": image_file()}, data={"mode": mode, "prompt": prompt}
    )
    assert response.status_code == 422
    assert fake.calls == []


def test_rejects_missing_inputs(client):
    session, fake = client
    assert session.post("/api/segment").status_code == 422
    assert fake.calls == []


@pytest.mark.parametrize(("payload", "code"), [(b"broken", 400), (b"a" * 1025, 413)])
def test_invalid_and_oversized_uploads(client, payload, code):
    session, fake = client
    response = session.post(
        "/api/segment",
        files={"image": ("image.png", payload)},
        data={"mode": "point", "prompt": "[0,0]"},
    )
    assert response.status_code == code
    assert fake.calls == []


def test_pixel_limit_checked_before_model(monkeypatch):
    monkeypatch.setenv("INSTASEG_MAX_IMAGE_PIXELS", "5")
    fake = FakeSegmenter()
    with TestClient(create_app(Settings.from_env(), fake)) as session:
        response = session.post(
            "/api/segment", files={"image": image_file()}, data={"mode": "point", "prompt": "[0,0]"}
        )
    assert response.status_code == 413
    assert not fake.calls


def test_exif_rotation_applied_before_coordinates(client):
    session, fake = client
    response = session.post(
        "/api/segment",
        files={"image": image_file(orientation=6)},
        data={"mode": "point", "prompt": "[2,3]"},
    )
    assert response.status_code == 200, response.text
    assert (response.json()["width"], response.json()["height"]) == (3, 4)
    assert fake.calls[0][1].shape == (4, 3, 3)
    assert decode_png(response.json()["mask"])[3, 2] == 255


@pytest.mark.parametrize("failure", ["fail", "bad_shape"])
def test_inference_errors_are_reported_without_internal_details(client, failure):
    session, fake = client
    setattr(fake, failure, True)
    response = session.post(
        "/api/segment", files={"image": image_file()}, data={"mode": "point", "prompt": "[0,0]"}
    )
    assert response.status_code == 500
    assert "private" not in response.text
    assert "模型推理失败" in response.json()["detail"]


def test_http_loads_model_once_on_startup(monkeypatch):
    fake = FakeSegmenter()
    loads = []

    def load(settings):
        loads.append(settings)
        return fake

    monkeypatch.setattr("instaseg.http_server.Sam21Segmenter.load", load)
    with TestClient(create_app()) as session:
        for _ in range(2):
            response = session.post(
                "/api/segment",
                files={"image": image_file()},
                data={"mode": "point", "prompt": "[0,0]"},
            )
            assert response.status_code == 200
    assert len(loads) == 1


def test_http_settings_use_separate_environment(monkeypatch):
    monkeypatch.setenv("INSTASEG_PORT", "50052")
    monkeypatch.setenv("INSTASEG_HTTP_PORT", "8080")
    assert Settings.from_env().port == 50052
    assert HttpSettings.from_env().port == 8080
    monkeypatch.setenv("INSTASEG_HTTP_PORT", "65536")
    with pytest.raises(ValueError, match="at most 65535"):
        HttpSettings.from_env()
