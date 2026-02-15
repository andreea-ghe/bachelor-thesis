#!/usr/bin/env python3
"""
Extract model weights from a PyTorch Lightning checkpoint.

This creates a weights-only file that can be used for fine-tuning
(starts fresh training instead of resuming from the saved epoch/optimizer).

Usage:
    python extract_weights.py checkpoint/jigsaw_4x4_128_512_250e_cosine_everyday.ckpt
    # Output: checkpoint/jigsaw_4x4_128_512_250e_cosine_everyday_weights.pt
"""
import sys
import os
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_weights.py <checkpoint_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckp = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in ckp:
        print("This checkpoint already contains only weights (no 'state_dict' key).")
        sys.exit(0)

    state_dict = ckp["state_dict"]
    print(f"  Epoch: {ckp.get('epoch', '?')}")
    print(f"  Parameters: {len(state_dict)} tensors")

    # Save weights only
    base, ext = os.path.splitext(ckpt_path)
    out_path = f"{base}_weights.pt"
    torch.save(state_dict, out_path)
    
    orig_size = os.path.getsize(ckpt_path) / 1024 / 1024
    new_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  Original: {orig_size:.1f} MB (full checkpoint with optimizer, scheduler, etc.)")
    print(f"  Extracted: {new_size:.1f} MB (weights only)")
    print(f"\n  Saved to: {out_path}")
    print(f"\n  Use this in your config: WEIGHT_FILE: {out_path}")


if __name__ == "__main__":
    main()

