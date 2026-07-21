import numpy as np
import pytest

from apairo.core.sample import Sample
from apairo_transform import to_tensors, to_torch_collate

torch = pytest.importorskip("torch")


def _s(**data):
    return Sample(data={k: np.asarray(v, dtype=np.float32) for k, v in data.items()})


class TestToTensors:
    def test_channels_become_tensors(self):
        out = to_tensors(_s(lidar=np.zeros((3, 4)), imu=np.ones(6)))
        assert set(out) == {"lidar", "imu"}
        assert isinstance(out["lidar"], torch.Tensor)
        assert out["lidar"].shape == (3, 4)


class TestToTorchCollate:
    def test_uniform_shapes_stack(self):
        batch = [_s(x=np.full((2, 3), i)) for i in range(4)]
        out = to_torch_collate(batch)
        assert isinstance(out["x"], torch.Tensor)
        assert out["x"].shape == (4, 2, 3)
        assert (out["x"][2] == 2).all()

    def test_ragged_channel_returns_list(self):
        # variable-length point clouds cannot dense-stack -> list of tensors
        batch = [_s(lidar=np.zeros((n, 4))) for n in (5, 8, 3)]
        out = to_torch_collate(batch)
        assert isinstance(out["lidar"], list)
        assert [t.shape[0] for t in out["lidar"]] == [5, 8, 3]

    def test_mixed_dense_and_ragged_keys(self):
        batch = [_s(lidar=np.zeros((n, 4)), imu=np.ones(6)) for n in (5, 8)]
        out = to_torch_collate(batch)
        assert isinstance(out["lidar"], list)  # ragged
        assert isinstance(out["imu"], torch.Tensor) and out["imu"].shape == (2, 6)

    def test_empty_batch(self):
        assert to_torch_collate([]) == {}
