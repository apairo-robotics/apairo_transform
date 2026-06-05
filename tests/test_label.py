import numpy as np
import pytest

from apairo_transform.label import RemapLabels, MaskLabels


def make_labels(values, dtype=np.int32) -> np.ndarray:
    return np.array(values, dtype=dtype)


# ── RemapLabels ───────────────────────────────────────────────────────────────

class TestRemapLabels:
    def test_basic_remap(self):
        labels = make_labels([0, 1, 2, 3])
        out = RemapLabels({0: 10, 1: 20, 2: 30, 3: 40})(labels)
        np.testing.assert_array_equal(out, [10, 20, 30, 40])

    def test_unknown_labels_get_default(self):
        labels = make_labels([0, 1, 99])
        out = RemapLabels({0: 0, 1: 1}, default=255)(labels)
        np.testing.assert_array_equal(out, [0, 1, 255])

    def test_custom_default(self):
        labels = make_labels([5, 6, 7])
        out = RemapLabels({5: 0}, default=0)(labels)
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_many_to_one(self):
        labels = make_labels([1, 2, 3, 4])
        out = RemapLabels({1: 0, 2: 0, 3: 1, 4: 1})(labels)
        np.testing.assert_array_equal(out, [0, 0, 1, 1])

    def test_2d_array(self):
        labels = make_labels([[0, 1], [2, 3]])
        out = RemapLabels({0: 10, 1: 20, 2: 30, 3: 40})(labels)
        np.testing.assert_array_equal(out, [[10, 20], [30, 40]])

    def test_output_dtype_matches_input(self):
        labels = make_labels([0, 1, 2], dtype=np.int64)
        out = RemapLabels({0: 5, 1: 6, 2: 7})(labels)
        assert out.dtype == labels.dtype

    def test_all_unknown(self):
        labels = make_labels([10, 20, 30])
        out = RemapLabels({0: 1}, default=255)(labels)
        np.testing.assert_array_equal(out, [255, 255, 255])

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError):
            RemapLabels({})

    def test_no_mutation_of_input(self):
        labels = make_labels([0, 1, 2])
        original = labels.copy()
        RemapLabels({0: 9, 1: 9, 2: 9})(labels)
        np.testing.assert_array_equal(labels, original)


# ── MaskLabels ────────────────────────────────────────────────────────────────

class TestMaskLabels:
    def test_keeps_specified_labels(self):
        labels = make_labels([0, 1, 2, 3])
        out = MaskLabels(keep={0, 1})(labels)
        np.testing.assert_array_equal(out, [0, 1, 255, 255])

    def test_custom_ignore_value(self):
        labels = make_labels([0, 1, 2])
        out = MaskLabels(keep={0}, ignore_value=-1)(labels)
        np.testing.assert_array_equal(out, [0, -1, -1])

    def test_keep_all(self):
        labels = make_labels([0, 1, 2, 3])
        out = MaskLabels(keep={0, 1, 2, 3})(labels)
        np.testing.assert_array_equal(out, labels)

    def test_mask_all(self):
        labels = make_labels([5, 6, 7])
        out = MaskLabels(keep={0, 1}, ignore_value=255)(labels)
        np.testing.assert_array_equal(out, [255, 255, 255])

    def test_2d_array(self):
        labels = make_labels([[0, 1], [2, 3]])
        out = MaskLabels(keep={0, 3})(labels)
        np.testing.assert_array_equal(out, [[0, 255], [255, 3]])

    def test_empty_keep_raises(self):
        with pytest.raises(ValueError):
            MaskLabels(keep=set())

    def test_no_mutation_of_input(self):
        labels = make_labels([0, 1, 2])
        original = labels.copy()
        MaskLabels(keep={0})(labels)
        np.testing.assert_array_equal(labels, original)

    def test_large_array(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 10, size=10_000).astype(np.int32)
        out = MaskLabels(keep={0, 1, 2})(labels)
        assert np.all((out == 255) | np.isin(out, [0, 1, 2]))
        assert np.all(out[np.isin(labels, [0, 1, 2])] == labels[np.isin(labels, [0, 1, 2])])
