"""Apply point-cloud transforms to a LiDAR channel at dataset access time.

Shows how :class:`RangeFilter`, :class:`RandomSubsample`, and
:class:`ShufflePoints` slot into a standard apairo dataset via
:meth:`~apairo.core.AbstractDataset.transform`.  No data is written to disk —
transforms are applied lazily on every ``__getitem__`` call.

Supported datasets: rellis, semantic_kitti.

Usage::

    python examples/pointcloud.py /data/RELLIS/00000
    python examples/pointcloud.py /data/sequences/00 --dataset semantic_kitti \\
        --max-range 50.0 --n-points 8192
"""

import argparse
from pathlib import Path

import numpy as np

from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_transform.pointcloud import RangeFilter, RandomSubsample, ShufflePoints

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
    p.add_argument("--max-range", type=float, default=50.0,
                   help="Keep only points within this range (metres).")
    p.add_argument("--n-points", type=int, default=8192,
                   help="Subsample to this many points after range filtering.")
    p.add_argument("--n-frames", type=int, default=5,
                   help="Number of frames to inspect.")
    args = p.parse_args()

    dataset_cls = DATASETS[args.dataset]
    ds = dataset_cls(args.seq_dir, keys=[args.lidar_key])

    ds.transform(args.lidar_key, RangeFilter(max=args.max_range))
    ds.transform(args.lidar_key, RandomSubsample(args.n_points))
    ds.transform(args.lidar_key, ShufflePoints())

    print(f"Dataset  : {dataset_cls.__name__}")
    print(f"Sequence : {args.seq_dir}")
    print(f"Frames   : {len(ds)}")
    print(f"Pipeline : RangeFilter(max={args.max_range}) → RandomSubsample({args.n_points}) → ShufflePoints()")
    print()

    for i in range(min(args.n_frames, len(ds))):
        sample = ds[i]
        pc = sample.data[args.lidar_key]
        ranges = np.linalg.norm(pc[:, :3], axis=1)
        print(
            f"  frame {i:04d} | pts={len(pc):<6} | "
            f"range [{ranges.min():.1f}, {ranges.max():.1f}] m"
        )


if __name__ == "__main__":
    main()
