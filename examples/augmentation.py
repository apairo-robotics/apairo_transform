"""Stochastic augmentation pipeline for LiDAR training data.

Builds a standard augmentation chain and applies it to frames of a sequence,
printing before/after statistics to verify the transforms are behaving.

Chain applied to ``lidar``:

  RangeFilter → RandomRotation(z) → RandomFlip(x) → RandomScale
              → RandomTranslation → GaussianNoise → RandomPointDrop

Supported datasets: rellis, semantic_kitti.

Usage::

    python examples/augmentation.py /data/RELLIS/00000
    python examples/augmentation.py /data/sequences/00 --dataset semantic_kitti \\
        --max-range 50.0 --rotation-axis so3
"""

import argparse
import math

import numpy as np

from apairo.core import Compose
from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_transform.pointcloud import (
    RangeFilter,
    RandomRotation,
    RandomFlip,
    RandomScale,
    RandomTranslation,
    GaussianNoise,
    RandomPointDrop,
)

DATASETS = {
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis")
    p.add_argument("--lidar-key", default="lidar")
    p.add_argument("--max-range", type=float, default=50.0)
    p.add_argument(
        "--rotation-axis",
        choices=["x", "y", "z", "so3"],
        default="z",
        help="Rotation axis. 'z' (yaw only) is standard for outdoor LiDAR.",
    )
    p.add_argument("--drop-p", type=float, default=0.05,
                   help="Per-point drop probability (default 0.05).")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dataset_cls = DATASETS[args.dataset]

    # Raw dataset — no transforms yet
    ds_raw = dataset_cls(args.seq_dir, keys=[args.lidar_key])

    # Augmented dataset
    ds_aug = dataset_cls(args.seq_dir, keys=[args.lidar_key])
    aug = Compose([
        RangeFilter(max=args.max_range),
        RandomRotation(axis=args.rotation_axis, seed=args.seed),
        RandomFlip(axis="x", p=0.5, seed=args.seed),
        RandomScale(range=(0.95, 1.05), seed=args.seed),
        RandomTranslation(sigma=0.2, seed=args.seed),
        GaussianNoise(sigma=0.01, seed=args.seed),
        RandomPointDrop(p=args.drop_p, seed=args.seed),
    ])
    ds_aug.transform(args.lidar_key, aug)

    print(f"Dataset   : {dataset_cls.__name__}")
    print(f"Sequence  : {args.seq_dir}")
    print(f"Pipeline  : {aug}")
    print()
    print(f"{'frame':<8} {'pts_raw':>8} {'pts_aug':>8}  {'centroid_raw (x,y,z)':>24}  {'centroid_aug (x,y,z)':>24}")
    print("─" * 82)

    for i in range(min(args.n_frames, len(ds_raw))):
        raw = ds_raw[i].data[args.lidar_key]
        aug_pc = ds_aug[i].data[args.lidar_key]

        c_raw = np.mean(raw[:, :3], axis=0)
        c_aug = np.mean(aug_pc[:, :3], axis=0) if len(aug_pc) > 0 else np.zeros(3)

        print(
            f"{i:04d}     {len(raw):>8} {len(aug_pc):>8}"
            f"  ({c_raw[0]:+6.2f} {c_raw[1]:+6.2f} {c_raw[2]:+6.2f})"
            f"  ({c_aug[0]:+6.2f} {c_aug[1]:+6.2f} {c_aug[2]:+6.2f})"
        )


if __name__ == "__main__":
    main()
