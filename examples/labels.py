"""Remap or mask semantic labels at dataset access time.

Shows how :class:`RemapLabels` and :class:`MaskLabels` integrate with apairo
datasets.  Transforms are applied lazily — no data is rewritten on disk.

The default mappings collapse Rellis-3D's 35 classes down to 5 coarse
categories (void / ground / vegetation / obstacle / sky) or keep only the
traversable subset.

Supported datasets: rellis, semantic_kitti.

Usage::

    python examples/labels.py /data/RELLIS/00000
    python examples/labels.py /data/sequences/00 --dataset semantic_kitti --mode mask
"""

import argparse
from collections import Counter

from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_transform.label import RemapLabels, MaskLabels

DATASETS = {
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}

# Rellis-3D: collapse to 5 coarse categories (0=void stays 0)
RELLIS_COARSE = {
    0: 0,   # void
    1: 1,   # dirt → ground
    3: 1,   # grass → ground
    4: 2,   # tree → vegetation
    5: 2,   # pole → obstacle
    6: 3,   # water → obstacle
    7: 2,   # vegetation → vegetation
    8: 3,   # obstacle → obstacle
    9: 4,   # sky → sky
    10: 1,  # asphalt → ground
    12: 3,  # building → obstacle
    15: 3,  # log → obstacle
    17: 3,  # person → obstacle
    18: 3,  # fence → obstacle
    19: 3,  # bush → vegetation
    23: 1,  # concrete → ground
    27: 3,  # barrier → obstacle
    31: 1,  # puddle → ground
    33: 1,  # mud → ground
    34: 3,  # rubble → obstacle
}

# SemanticKITTI: keep road + terrain + vegetation; ignore everything else
SEMANTIC_KITTI_KEEP = {40, 44, 48, 49, 60, 70, 71, 72}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis")
    p.add_argument("--labels-key", default="labels")
    p.add_argument(
        "--mode",
        choices=["remap", "mask"],
        default="remap",
        help="'remap' collapses classes; 'mask' ignores all but a kept subset.",
    )
    p.add_argument("--n-frames", type=int, default=5)
    args = p.parse_args()

    dataset_cls = DATASETS[args.dataset]
    ds = dataset_cls(args.seq_dir, keys=[args.labels_key])

    if args.mode == "remap":
        mapping = RELLIS_COARSE if args.dataset == "rellis" else {
            k: 0 for k in range(260)
        }
        ds.transform(args.labels_key, RemapLabels(mapping))
        desc = f"RemapLabels ({len(mapping)} rules)"
    else:
        keep = {1, 3, 10, 23, 31, 33} if args.dataset == "rellis" else SEMANTIC_KITTI_KEEP
        ds.transform(args.labels_key, MaskLabels(keep=keep, ignore_value=255))
        desc = f"MaskLabels(keep={sorted(keep)})"

    print(f"Dataset   : {dataset_cls.__name__}")
    print(f"Sequence  : {args.seq_dir}")
    print(f"Transform : {desc}")
    print()

    for i in range(min(args.n_frames, len(ds))):
        sample = ds[i]
        labels = sample.data[args.labels_key]
        counts = Counter(labels.tolist())
        unique = sorted(counts)
        print(f"  frame {i:04d} | unique classes: {unique}")


if __name__ == "__main__":
    main()
