"""AccumulateFrames is deprecated in favour of window(reduce=StackByPose())."""

import pytest

from apairo_transform import AccumulateFrames, StackByPose


def test_warns_and_names_its_replacement():
    with pytest.warns(DeprecationWarning, match=r"window\(.*StackByPose") as record:
        AccumulateFrames(num_frames=3)
    assert record[0].filename == __file__


def test_still_validates_its_arguments():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        AccumulateFrames(num_frames=0)


def test_replacement_is_exported():
    assert callable(StackByPose())
