#!/usr/bin/env python3
"""
Split a fracture pairs CSV into train/val/test files at the OBJECT level.

The split is done at the object level (not pair level) to prevent data leakage:
all fracture patterns of the same object go into the same split.

Usage:
    python create_pair_splits.py \
        --csv /home/dataset/everyday_all_2/everyday_pairs_all_2.csv \
        --output_dir /home/dataset/everyday_all_2/ \
        --seed 42

    # For artifact:
    python create_pair_splits.py \
        --csv /home/dataset/artifact_all_2/artifact_pairs_all_2.csv \
        --subset artifact \
        --output_dir /home/dataset/artifact_all_2/

    # Custom ratios:
    python create_pair_splits.py \
        --csv /home/dataset/everyday_all_2/everyday_pairs_all_2.csv \
        --output_dir /home/dataset/everyday_all_2/ \
        --train_ratio 0.8 --val_ratio 0.1
"""
import argparse
import csv
import os
import random
from collections import defaultdict


def piece_name_to_object_key(piece_name):
    """
    Extract the object identity from a piece filename.
    
    'everyday_BeerBottle_hash_fractured_0_piece_0.obj'
    -> 'everyday_BeerBottle_hash'
    
    'artifact_73400_sf_fractured_0_piece_0.obj'
    -> 'artifact_73400_sf'
    """
    name = piece_name.replace(".obj", "")
    parts = name.split("_fractured_")
    return parts[0]


def write_split_file(filepath, pairs):
    """Write a list of (pc1, pc2) pairs to a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc1", "pc2"])
        for pc1, pc2 in pairs:
            writer.writerow([pc1, pc2])


def main():
    parser = argparse.ArgumentParser(
        description="Split fracture pairs CSV into train/val/test at the object level"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to pairs CSV file (with pc1,pc2 columns)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write split files")
    parser.add_argument("--subset", type=str, default="everyday",
                        help="Dataset subset: 'everyday' or 'artifact'")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of objects for training (default 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of objects for validation (default 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print(f"ERROR: train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) > 1.0")
        return

    print("=" * 70)
    print("  Fracture Pairs Split Creator")
    print("=" * 70)
    print(f"  CSV file     : {args.csv}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Subset       : {args.subset}")
    print(f"  Ratios       : train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"  Seed         : {args.seed}")
    print("=" * 70)

    # ── Step 1: Read all pairs from CSV ──────────────────────────────────
    print("\n[Step 1] Reading pairs from CSV...")
    pairs = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["pc1"].strip(), row["pc2"].strip()))
    print(f"  Total pairs: {len(pairs)}")

    # ── Step 2: Group pairs by object ────────────────────────────────────
    print("\n[Step 2] Grouping pairs by object...")
    object_pairs = defaultdict(list)
    for pc1, pc2 in pairs:
        obj_key = piece_name_to_object_key(pc1)
        object_pairs[obj_key].append((pc1, pc2))

    print(f"  Unique objects: {len(object_pairs)}")
    total_from_groups = sum(len(v) for v in object_pairs.values())
    print(f"  Total pairs from groups: {total_from_groups}")

    counts = [len(v) for v in object_pairs.values()]
    print(f"  Pairs per object: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")

    # ── Step 3: Split objects into train/val/test ────────────────────────
    print("\n[Step 3] Splitting objects...")
    all_keys = sorted(object_pairs.keys())
    random.seed(args.seed)
    random.shuffle(all_keys)

    n = len(all_keys)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_obj_keys = set(all_keys[:train_end])
    val_obj_keys = set(all_keys[train_end:val_end])
    test_obj_keys = set(all_keys[val_end:])

    # ── Step 4: Assign pairs to splits ───────────────────────────────────
    train_pairs = []
    val_pairs = []
    test_pairs = []
    for obj_key, obj_pairs in object_pairs.items():
        if obj_key in train_obj_keys:
            train_pairs.extend(obj_pairs)
        elif obj_key in val_obj_keys:
            val_pairs.extend(obj_pairs)
        else:
            test_pairs.extend(obj_pairs)

    print(f"\n  Split results:")
    print(f"    Train: {len(train_obj_keys):>4} objects, {len(train_pairs):>5} pairs")
    print(f"    Val  : {len(val_obj_keys):>4} objects, {len(val_pairs):>5} pairs")
    print(f"    Test : {len(test_obj_keys):>4} objects, {len(test_pairs):>5} pairs")
    total_obj = len(train_obj_keys) + len(val_obj_keys) + len(test_obj_keys)
    total_p = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"    Total: {total_obj:>4} objects, {total_p:>5} pairs")

    # ── Step 5: Write output files ───────────────────────────────────────
    print(f"\n[Step 4] Writing split files...")
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.output_dir, f"{args.subset}.train.txt")
    val_file = os.path.join(args.output_dir, f"{args.subset}.val.txt")
    test_file = os.path.join(args.output_dir, f"{args.subset}.test.txt")

    write_split_file(train_file, train_pairs)
    print(f"  Written: {train_file} ({len(train_pairs)} pairs)")

    write_split_file(val_file, val_pairs)
    print(f"  Written: {val_file} ({len(val_pairs)} pairs)")

    write_split_file(test_file, test_pairs)
    print(f"  Written: {test_file} ({len(test_pairs)} pairs)")

    print("\nDone!")


if __name__ == "__main__":
    main()
