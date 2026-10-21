from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from instaseg.predictor import Sam21Segmenter


@pytest.mark.parametrize("mode", ["point", "box"])
def test_prompt_forwarding_and_best_mask_selection(monkeypatch, mode):
    # Exercise the real adapter without installing torch or downloading weights.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    prompts = []
    images = []

    def predict(**kwargs):
        prompts.append(kwargs)
        return np.stack([np.zeros((3, 4)), np.ones((3, 4))]), np.array([0.1, 0.9]), None

    predictor = SimpleNamespace(set_image=images.append, predict=predict)
    segmenter = Sam21Segmenter(predictor, "cpu")
    if mode == "point":
        result = segmenter.segment(image, 2, 1)
        np.testing.assert_array_equal(prompts[0]["point_coords"], [[2, 1]])
        np.testing.assert_array_equal(prompts[0]["point_labels"], [1])
        assert "box" not in prompts[0]
    else:
        result = segmenter.segment_box(image, (0, 0, 4, 3))
        np.testing.assert_array_equal(prompts[0]["box"], [0, 0, 4, 3])
        assert "point_coords" not in prompts[0]
    assert prompts[0]["multimask_output"] is True
    assert images[0] is image
    assert result.dtype == np.bool_
    assert result.all()
