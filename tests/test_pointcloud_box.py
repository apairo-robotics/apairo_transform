import numpy as np

from apairo_transform.pointcloud import BoxFilter


def test_xy_box_keeps_inside():
    pc = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = BoxFilter(x_min=-1.0, x_max=10.0, y_min=-1.0, y_max=10.0)(pc)
    np.testing.assert_array_equal(out, pc[[0, 1]])


def test_z_bounds_optional():
    pc = np.array([[0, 0, -1.0], [0, 0, 0.5], [0, 0, 3.0]], dtype=np.float32)
    out = BoxFilter(z_min=0.0, z_max=1.0)(pc)
    np.testing.assert_array_equal(out, pc[[1]])


def test_no_bounds_keeps_all():
    pc = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    np.testing.assert_array_equal(BoxFilter()(pc), pc)


def test_compute_mask_alignment():
    pc = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
    mask = BoxFilter(x_max=10.0)(pc)
    assert len(mask) == 1
    m = BoxFilter(x_max=10.0).compute_mask(pc)
    assert m.tolist() == [True, False]
