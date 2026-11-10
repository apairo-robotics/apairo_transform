# Changelog

## [0.3.1] - 2026-11-10

A code review of 0.3.0. The geometry was checked against scipy and is right
(quaternion order, transform direction, slerp, stacking by pose); the defects
were in randomness, array types and one example.

### Fixed

- **Random transforms no longer repeat themselves across `DataLoader`
  workers.** Each built its generator in the constructor, so every worker held
  a copy of the same state: two workers drew the same "random" rotations, and
  every epoch replayed them, with or without a seed. The generator is now
  created where it is used, once per process and per worker epoch. With a
  seed, the main process draws what it drew before; inside a worker the seed
  is combined with the worker's own, so a run under `torch.manual_seed` is
  reproducible while workers and epochs differ. Affects `RandomRotation`,
  `RandomFlip`, `RandomScale`, `RandomTranslation`, `GaussianNoise`,
  `RandomPointDrop`, `RandomSubsample` and `ShufflePoints`.
- **`Resize` keeps the image's dtype.** It cast its result to `uint8`: a float
  image in [0, 1] became all zeros, a 16-bit depth wrapped modulo 256. Its
  docstring now says `width` x `height`, and that label images need
  `method="nearest"`.
- **`RemapLabels` and `MaskLabels` hold the values asked for.** A target or
  ignore value that did not fit the labels' type wrapped under numpy 1.26 (300
  written as 44 in `uint8`, 255 as -1 in `int8`) and raised under numpy 2. The
  result is widened to the next integer type that holds them.
- **`RandomSubsample`, `CastTo` and `RemapLabels` can be saved in a deployed
  chain** (`ds.inference(...).save()`), like the other transforms kept at
  inference.
- A zero quaternion (an uninitialised pose) raises an error naming the pose,
  in `PoseTo4x4` and `Se3Interp`; it gave a NaN rotation.
- `VoxelDownsample` drops points without a position (NaN or infinite xyz);
  each became a voxel with a NaN centroid. `reduction="random"` takes a
  `seed`.
- `TransformPoints` keeps its own copy of the matrix.
- `examples/compose.py` filtered and subsampled the cloud but not its
  per-point labels, so label `i` no longer described point `i`. It now applies
  one selection to both, and the README says why (*Points with per-point
  labels*).

### Changed

- **`RangeFilter` has no upper bound by default.** `max` defaulted to 50.0, so
  `RangeFilter(min=1.0)` also removed everything beyond 50 m and
  `RangeFilter(min=60.0)` failed. Pass `max=50.0` to keep the old behaviour.
- `RangeFilter` and `HeightFilter` are documented as `[min, max)`, which is
  what they do; they were documented as `[min, max]`.

## [0.3.0] - 2026-11-09

First release on PyPI.
