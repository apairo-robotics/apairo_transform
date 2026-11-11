"""Full preprocessing pipeline using Compose on multiple channels.

Shows how to build a training-ready sample in one pass:

  1. points and their labels together: range filter, then subsample -- one
     selection applied to both, so label ``i`` still describes point ``i``;
  2. ``lidar``  : ChannelSelect (xyz only)
  3. ``labels`` : RemapLabels (collapse to coarse classes) → MaskLabels (ignore void)

A transform registered on one channel sees that channel only. A step that
drops or reorders points (RangeFilter, RandomSubsample, RandomPointDrop,
ShufflePoints, VoxelDownsample) therefore cannot be registered on ``lidar``
alone when the sample also carries per-point labels: the cloud would lose
points and the labels would not. Such a step is written at sample level, as
``select_points`` below does.

:class:`~apairo.core.Compose` from apairo core chains the per-channel steps.
All transforms are applied lazily — no data is written to disk.

Supported datasets: rellis, semantic_kitti.

Usage::

    python examples/compose.py /data/RELLIS/00000
    python examples/compose.py /data/sequences/00 --dataset semantic_kitti \\
        --max-range 50.0 --n-points 16384
"""

import argparse

import numpy as np

from apairo.core import Compose
from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_transform._random import Rng
from apairo_transform.pointcloud import ChannelSelect, RangeFilter
from apairo_transform.label import MaskLabels, RemapLabels

DATASETS = {
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}

# Rellis-3D: 35 → 5 coarse classes
RELLIS_COARSE = {
    1: 1, 3: 1, 10: 1, 23: 1, 31: 1, 33: 1,   # ground
    4: 2, 7: 2, 19: 2,                          # vegetation
    5: 3, 6: 3, 8: 3, 12: 3, 15: 3,
    17: 3, 18: 3, 27: 3, 34: 3,                 # obstacle
    9: 4,                                        # sky
}

SEMANTIC_KITTI_COARSE = {
    40: 1, 44: 1, 48: 1, 49: 1, 60: 1,         # ground
    70: 2, 71: 2, 72: 2,                         # vegetation
    50: 3, 51: 3, 52: 3, 80: 3, 81: 3,          # moving objects
    11: 4, 15: 4, 16: 4,                         # structure
}


def select_points(lidar_key: str, labels_key: str, max_range: float, n_points: int,
                  seed: int | None = None):
    """A sample-level step: keep the points within *max_range*, at most
    *n_points* of them, and the labels of exactly those points."""
    in_range = RangeFilter(max=max_range)
    rng = Rng(seed)

    def step(sample):
        points, labels = sample.data[lidar_key], sample.data[labels_key]
        keep = np.flatnonzero(in_range.compute_mask(points))
        if len(keep) > n_points:
            keep = np.sort(rng.choice(keep, n_points, replace=False))
        sample.data[lidar_key] = points[keep]
        sample.data[labels_key] = labels[keep]
        return sample

    return step


def build_pipeline(dataset: str):
    """Return (lidar_transform, labels_transform) for the chosen dataset: the
    per-channel steps, which keep every point."""
    lidar_tf = Compose([
        ChannelSelect([0, 1, 2]),   # drop intensity, keep xyz
    ])

    coarse = RELLIS_COARSE if dataset == "rellis" else SEMANTIC_KITTI_COARSE
    labels_tf = Compose([
        RemapLabels(coarse, default=0),
        MaskLabels(keep=set(coarse.values()), ignore_value=255),
    ])

    return lidar_tf, labels_tf


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis")
    p.add_argument("--lidar-key", default="lidar")
    p.add_argument("--labels-key", default="labels")
    p.add_argument("--max-range", type=float, default=50.0)
    p.add_argument("--n-points", type=int, default=8192)
    p.add_argument("--n-frames", type=int, default=5)
    args = p.parse_args()

    dataset_cls = DATASETS[args.dataset]
    ds = dataset_cls(args.seq_dir, keys=[args.lidar_key, args.labels_key])

    lidar_tf, labels_tf = build_pipeline(args.dataset)
    points_step = select_points(
        args.lidar_key, args.labels_key, args.max_range, args.n_points
    )

    ds = ds.transform(points_step, in_place=False)   # points and labels together
    ds = ds.transform(args.lidar_key, lidar_tf, in_place=False)
    ds = ds.transform(args.labels_key, labels_tf, in_place=False)

    print(f"Dataset   : {dataset_cls.__name__}")
    print(f"Sequence  : {args.seq_dir}")
    print(f"lidar     : {lidar_tf}")
    print(f"labels    : {labels_tf}")
    print()

    for i in range(min(args.n_frames, len(ds))):
        sample = ds[i]
        xyz = sample.data[args.lidar_key]
        lbl = sample.data[args.labels_key]
        valid = lbl[lbl != 255]
        print(
            f"  frame {i:04d} | pts={len(xyz):<6} xyz  | "
            f"labels unique={sorted(set(valid.tolist()))}  "
            f"({(lbl == 255).sum()} ignored)"
        )


if __name__ == "__main__":
    main()
