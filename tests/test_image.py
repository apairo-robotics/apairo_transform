import numpy as np
import pytest

from apairo_transform.image import CenterCrop, ToFloat, Normalize, ToCHW

IMG = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)


# ── CenterCrop (pure numpy) ─────────────────────────────────────────────────

class TestCenterCrop:
    def test_shape(self):
        out = CenterCrop(24, 32)(IMG)
        assert out.shape == (24, 32, 3)

    def test_centred(self):
        out = CenterCrop(24, 32)(IMG)
        np.testing.assert_array_equal(out, IMG[12:36, 16:48])

    def test_larger_than_image_clamps(self):
        out = CenterCrop(100, 100)(IMG)
        assert out.shape == (48, 64, 3)


# ── ToFloat / Normalize / ToCHW ─────────────────────────────────────────────

class TestToFloat:
    def test_range(self):
        out = ToFloat()(IMG)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_values(self):
        out = ToFloat()(np.array([[[0, 128, 255]]], dtype=np.uint8))
        np.testing.assert_allclose(out.ravel(), [0.0, 128 / 255, 1.0], atol=1e-6)


class TestNormalize:
    def test_imagenet_default(self):
        f = ToFloat()(IMG)
        out = Normalize()(f)
        # channel 0 mean should shift by -0.485 / 0.229 roughly
        assert out.shape == f.shape
        assert out.dtype == np.float32

    def test_custom_stats_zero(self):
        f = np.full((2, 2, 3), 0.5, dtype=np.float32)
        out = Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))(f)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


class TestToCHW:
    def test_transpose(self):
        out = ToCHW()(IMG)
        assert out.shape == (3, 48, 64)
        assert out.flags["C_CONTIGUOUS"]


# ── Resize (needs OpenCV) ───────────────────────────────────────────────────

class TestResize:
    def test_unknown_method_raises(self):
        from apairo_transform.image import Resize

        with pytest.raises(ValueError):
            Resize(32, 24, method="bogus")

    def test_resize_shape(self):
        cv2 = pytest.importorskip("cv2")
        from apairo_transform.image import Resize

        out = Resize(32, 24, method="area")(IMG)
        assert out.shape == (24, 32, 3)
        assert out.dtype == np.uint8
