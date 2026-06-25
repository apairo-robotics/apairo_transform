# apairo-transform

Runtime transforms and augmentations for [apairo](https://github.com/apairo-robotics/apairo) datasets — applied lazily at access time, with no data written to disk.

Unlike [apairo-preprocess](https://github.com/apairo-robotics/apairo_preprocess) (offline, disk-bound), transforms are registered on a dataset and executed on every `__getitem__` call.

---

## Installation

```bash
pip install git+https://github.com/apairo-robotics/apairo_transform.git
```

Requires Python ≥ 3.11 and numpy. No other hard dependencies.

---

## Transforms

### Point cloud — filter & downsample

| Class | Description |
|---|---|
| `RangeFilter(min, max)` | Keep points whose distance from the origin is within `[min, max]` metres |
| `RandomSubsample(n)` | Randomly subsample to at most `n` points |
| `ShufflePoints()` | Randomly permute the point order |
| `ChannelSelect(channels)` | Select a subset of columns (e.g. `[0,1,2]` for xyz only) |
| `VoxelDownsample(voxel_size)` | In-memory voxel-grid downsampling — one representative point per voxel |

### Point cloud — temporal accumulation

| Class | Description |
|---|---|
| `AccumulateFrames(lidar, pose, num_frames, stride)` | Densify the current scan by stacking the previous `num_frames` frames (every `stride`-th), each placed into the current ego frame via its pose |

Unlike the other point-cloud transforms, this one reads **two** channels (the
scan *and* the pose), so it is registered as a *sample-level* transform and
keeps the frames it has seen in an internal rolling buffer. It is therefore
**stateful and order-dependent** — accumulate on a sequential pass *before* any
shuffle/cache, and call `.reset()` between epochs.

```python
from apairo_transform import AccumulateFrames

ds.transform(AccumulateFrames(lidar="lidar", pose="pose", num_frames=5, stride=1))
```

### Point cloud — augmentation

| Class | Description |
|---|---|
| `RandomRotation(axis, max_angle)` | Rotate xyz by a random angle around `"x"`, `"y"`, `"z"`, or full `"so3"` |
| `RandomFlip(axis, p)` | Mirror along `"x"` or `"y"` with probability `p` |
| `RandomScale(range)` | Scale xyz uniformly by a random factor in `range` |
| `RandomTranslation(sigma)` | Translate all points by a random vector drawn from N(0, σ²) |
| `GaussianNoise(sigma)` | Add independent per-point Gaussian noise to xyz |
| `RandomPointDrop(p)` | Drop each point independently with probability `p` |

### Pose

| Class | Description |
|---|---|
| `PoseTo4x4()` | Convert `(N,7)` [tx ty tz qx qy qz qw] or `(N,6)` [tx ty tz rx ry rz] to `(N,4,4)` |
| `InvertPose()` | Closed-form rigid inverse of a `(4,4)` or `(N,4,4)` homogeneous matrix |

### Label

| Class | Description |
|---|---|
| `RemapLabels(mapping)` | Remap integer class IDs via a `{old: new}` dictionary |
| `MaskLabels(keep, ignore_value)` | Set any label not in `keep` to `ignore_value` (default 255) |

### Interpolation — for `ds.synchronize()`

Value-level strategies implementing the `apairo.Interpolator` contract:
instead of matching an existing event, the value is *synthesized* at the
reference instant from its two bracketing events. Continuous signals only.

| Class | Description |
|---|---|
| `LinearInterp()` | Linear blend between the bracketing values (commands, IMU, scalars) |
| `Se3Interp()` | SE(3) pose interpolation — lerp translation + shortest-path slerp rotation; accepts `(7,)` [tx ty tz qx qy qz qw] or `(4,4)` |

```python
from apairo_transform.interp import LinearInterp, Se3Interp

ds_sync = ds.synchronize(
    reference="velodyne_0",
    method={
        "gicp_poses": Se3Interp(),     # pose interpolated at each lidar instant
        "cmd":        LinearInterp(),
    },                                  # unlisted channels -> "latest"
)
```

---

## Quickstart

### Registering transforms on a dataset

Transforms are registered on a dataset channel and applied lazily on every `__getitem__`.

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_transform import RangeFilter, RandomSubsample, ChannelSelect

ds = Rellis3DDataset("/data/RELLIS/00000", keys=["lidar"])

ds.transform("lidar", RangeFilter(max=50.0))
ds.transform("lidar", RandomSubsample(8192))
ds.transform("lidar", ChannelSelect([0, 1, 2]))   # xyz only

sample = ds[0]
# sample.data["lidar"] -> np.ndarray (N, 3), N ≤ 8192, range < 50 m
```

Multiple calls on the same key compose in order. Alternatively, use [`Compose`](https://github.com/apairo-robotics/apairo) from apairo core:

```python
from apairo.core import Compose
from apairo_transform import RangeFilter, RandomSubsample, ChannelSelect

ds.transform("lidar", Compose([
    RangeFilter(max=50.0),
    RandomSubsample(8192),
    ChannelSelect([0, 1, 2]),
]))
```

### Augmentation pipeline for training

```python
from apairo.core import Compose
from apairo.dataset.rellis import Rellis3DDataset
from apairo_transform import (
    RangeFilter, RandomRotation, RandomFlip,
    RandomScale, GaussianNoise, RandomPointDrop,
)

ds = Rellis3DDataset("/data/RELLIS/00000", keys=["lidar"])

ds.transform("lidar", Compose([
    RangeFilter(max=50.0),
    RandomRotation(axis="z"),           # yaw-only for outdoor LiDAR
    RandomFlip(axis="x", p=0.5),
    RandomScale(range=(0.95, 1.05)),
    GaussianNoise(sigma=0.01),
    RandomPointDrop(p=0.05),
]))
```

### Label remapping

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_transform import RemapLabels, MaskLabels

ds = Rellis3DDataset("/data/RELLIS/00000", keys=["labels"])

# Collapse 35 Rellis classes to 4 coarse categories
COARSE = {1: 0, 3: 0, 10: 0, 23: 0,   # ground
          4: 1, 7: 1, 19: 1,            # vegetation
          8: 2, 12: 2, 15: 2, 18: 2,   # obstacle
          9: 3}                          # sky

ds.transform("labels", RemapLabels(COARSE, default=255))
ds.transform("labels", MaskLabels(keep={0, 1, 2, 3}, ignore_value=255))
```

### Pose conversion

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_transform import PoseTo4x4, InvertPose

ds = Rellis3DDataset("/data/RELLIS/00000", keys=["poses"])

# poses channel: (3, 4) → pad to (4, 4) and invert
ds.transform("poses", PoseTo4x4())
ds.transform("poses", InvertPose())
```

### Voxel downsampling at access time

Unlike `VoxelisePointCloud` in apairo-preprocess (which saves to disk once), `VoxelDownsample` runs in memory on every access — useful for dynamic voxel sizes or when disk space is constrained.

```python
from apairo_transform import VoxelDownsample

ds.transform("lidar", VoxelDownsample(voxel_size=0.1, max_range=50.0))
```

---

## Examples

Ready-to-run scripts in [`examples/`](examples/):

```bash
# Filter + subsample a LiDAR channel
python examples/pointcloud.py /data/RELLIS/00000

# Remap or mask semantic labels
python examples/labels.py /data/RELLIS/00000 --mode remap

# Full multi-channel pipeline with Compose
python examples/compose.py /data/RELLIS/00000

# Augmentation pipeline — prints before/after statistics per frame
python examples/augmentation.py /data/RELLIS/00000 --rotation-axis z --drop-p 0.05
```

---

## Relationship to apairo-preprocess

| | apairo-transform | apairo-preprocess |
|---|---|---|
| **When** | At access time (lazy) | Offline, once |
| **Output** | In-memory array | Files on disk |
| **Deps** | numpy only | numpy, scipy, KISS-ICP, … |
| **Use case** | Training loops, augmentation | Dataset preparation |

Both packages are complementary: run apairo-preprocess once to produce derived channels (odometry, traversability, voxelised coordinates), then use apairo-transform at training time to augment and normalise the data.

---

## License

MIT
