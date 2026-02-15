#!/usr/bin/env python3
"""
Split a fracture pairs CSV into train/test files at the OBJECT level.

The split is done at the object level (not pair level) to prevent data leakage:
all fracture patterns of the same object go into the same split.

Usage:
    # Random split:
    python create_pair_splits.py \
        --csv /home/dataset/everyday_all_2/everyday_pairs_all_2.csv \
        --output_dir /home/dataset/everyday_all_2/ \
        --seed 42

    # For artifact:
    python create_pair_splits.py \
        --csv /home/dataset/artifact_all_2/artifact_pairs_all_2.csv \
        --subset artifact \
        --output_dir /home/dataset/artifact_all_2/
"""
import argparse
import csv
import os
import random
from collections import defaultdict
from typing import Any


# Known categories for object path reconstruction
EVERYDAY_CATEGORIES = [
    "BeerBottle", "Bowl", "Cup", "DrinkingUtensil", "Mug", "Plate",
    "Spoon", "Teacup", "ToyFigure", "WineBottle", "Bottle", "Cookie",
    "DrinkBottle", "Mirror", "PillBottle", "Ring", "Statue", "Teapot",
    "Vase", "WineGlass",
]

# Sorted longest-first so "DrinkBottle" matches before "Bottle"
EVERYDAY_CATEGORIES_SORTED = sorted(EVERYDAY_CATEGORIES, key=len, reverse=True)


def piece_name_to_object_key(piece_name):
    """
    Extract the object identity from a piece filename.
    
    'everyday_BeerBottle_hash_fractured_0_piece_0.obj'
    -> 'everyday_BeerBottle_hash'
    
    'artifact_73400_sf_fractured_0_piece_0.obj'
    -> 'artifact_73400_sf'
    """
    # Remove .obj extension if present
    name = piece_name.replace(".obj", "")
    # Split on '_fractured_' to separate object from fracture pattern
    parts = name.split("_fractured_")
    return parts[0]


def main():
    parser = argparse.ArgumentParser(
        description="Split fracture pairs CSV into train/test at the object level"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to pairs CSV file (with pc1,pc2 columns)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write everyday.train.txt and everyday.test.txt")
    parser.add_argument("--subset", type=str, default="everyday",
                        help="Dataset subset: 'everyday' or 'artifact'")
    parser.add_argument("--train_ratio", type=float, default=0.82,
                        help="Train ratio if no original splits (default 0.82 = 407/498)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    args = parser.parse_args()

    print("=" * 70)
    print("  Fracture Pairs Split Creator")
    print("=" * 70)
    print(f"  CSV file    : {args.csv}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Subset      : {args.subset}")
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

    # Show distribution
    counts = [len(v) for v in object_pairs.values()]
    print(f"  Pairs per object: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")

    # ── Step 3: Determine train/test split ───────────────────────────────
    print("\n[Step 3] Determining train/test split...")
    
    train_obj_keys = set()
    test_obj_keys = set[Any]()

    # Random split
    all_keys = sorted(object_pairs.keys())
    random.seed(args.seed)
    random.shuffle(all_keys)
    split_idx = int(len(all_keys) * args.train_ratio)
    train_obj_keys = set(all_keys[:split_idx])
    test_obj_keys = set(all_keys[split_idx:])

    # ── Step 4: Create train/test pair lists ─────────────────────────────
    train_pairs = []
    test_pairs = []
    for obj_key, obj_pairs in object_pairs.items():
        if obj_key in train_obj_keys:
            train_pairs.extend(obj_pairs)
        elif obj_key in test_obj_keys:
            test_pairs.extend(obj_pairs)

    print(f"\n  Split results:")
    print(f"    Train: {len(train_obj_keys)} objects, {len(train_pairs)} pairs")
    print(f"    Test : {len(test_obj_keys)} objects, {len(test_pairs)} pairs")
    print(f"    Total: {len(train_obj_keys) + len(test_obj_keys)} objects, {len(train_pairs) + len(test_pairs)} pairs")

    # ── Step 5: Write output files ───────────────────────────────────────
    print(f"\n[Step 4] Writing split files...")
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.output_dir, f"{args.subset}.train.txt")
    test_file = os.path.join(args.output_dir, f"{args.subset}.test.txt")

    with open(train_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc1", "pc2"])
        for pc1, pc2 in train_pairs:
            writer.writerow([pc1, pc2])
    print(f"  Written: {train_file} ({len(train_pairs)} pairs)")

    with open(test_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc1", "pc2"])
        for pc1, pc2 in test_pairs:
            writer.writerow([pc1, pc2])
    print(f"  Written: {test_file} ({len(test_pairs)} pairs)")

    print("\nDone!")


if __name__ == "__main__":
    main()
