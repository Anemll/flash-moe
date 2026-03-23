#!/usr/bin/env python3
"""
Create a smoke-test subset of the 2-bit expert model for iPhone testing.

Takes the first N experts (default 32) from each layer file, creating a
much smaller model (~7 GB instead of 120 GB) that can be transferred to
an iPhone for bringup testing.

Dense weights (model_weights.bin, vocab.bin, tokenizer.bin) are symlinked
from the source model. A config.json override limits num_experts so the
routing gate only scores the available experts.

Usage:
    python3 create_smoke_model.py \
        --model ~/Models/flash_mlx_4bit \
        --output ~/Models/flash_mlx_4bit_smoke32 \
        --num-experts 32

    # Then run:
    ./infer --model ~/Models/flash_mlx_4bit_smoke32 --2bit \
        --prompt "Hello" --tokens 20 --stream
"""

import argparse
import json
import os
import shutil
import sys

def main():
    parser = argparse.ArgumentParser(description="Create smoke-test expert subset")
    parser.add_argument("--model", required=True, help="Source model directory (e.g. ~/Models/flash_mlx_4bit)")
    parser.add_argument("--output", required=True, help="Output directory for smoke model")
    parser.add_argument("--num-experts", type=int, default=32, help="Number of experts to keep per layer (default: 32)")
    parser.add_argument("--num-layers", type=int, default=60, help="Number of layers (default: 60)")
    parser.add_argument("--source-experts", default="packed_experts_2bit",
                        help="Source expert directory name (default: packed_experts_2bit)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model)
    out_dir = os.path.expanduser(args.output)
    src_expert_dir = os.path.join(model_dir, args.source_experts)
    N = args.num_experts

    # Validate source
    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(src_expert_dir):
        print(f"ERROR: Expert directory not found: {src_expert_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect expert size from first layer file
    layer0 = os.path.join(src_expert_dir, "layer_00.bin")
    if not os.path.isfile(layer0):
        print(f"ERROR: {layer0} not found", file=sys.stderr)
        sys.exit(1)

    layer_file_size = os.path.getsize(layer0)
    full_num_experts = 512
    expert_size = layer_file_size // full_num_experts
    if layer_file_size != expert_size * full_num_experts:
        print(f"WARNING: layer file size {layer_file_size} not evenly divisible by {full_num_experts}")

    subset_layer_size = expert_size * N
    total_size_gb = subset_layer_size * args.num_layers / (1024**3)

    print(f"Source: {model_dir}")
    print(f"Expert dir: {args.source_experts}")
    print(f"Expert size: {expert_size} bytes ({expert_size/1024/1024:.2f} MB)")
    print(f"Keeping {N}/{full_num_experts} experts per layer")
    print(f"Per layer: {subset_layer_size/1024/1024:.1f} MB (was {layer_file_size/1024/1024:.1f} MB)")
    print(f"Total experts: {total_size_gb:.2f} GB (was {layer_file_size * args.num_layers / 1024**3:.1f} GB)")
    print()

    # Create output
    if os.path.exists(out_dir) and not args.force:
        print(f"ERROR: Output exists: {out_dir} (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # Symlink large dense files, patch model_weights.json
    symlink_files = [
        "model_weights.bin",
        "vocab.bin",
        "tokenizer.bin",
    ]
    for f in symlink_files:
        src = os.path.join(model_dir, f)
        dst = os.path.join(out_dir, f)
        if os.path.isfile(src):
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            print(f"  symlink: {f}")
        else:
            print(f"  skip (not found): {f}")

    # Patch model_weights.json: set num_experts to N
    # (manifest loads after config.json and would override it)
    manifest_src = os.path.join(model_dir, "model_weights.json")
    manifest_dst = os.path.join(out_dir, "model_weights.json")
    if os.path.isfile(manifest_src):
        with open(manifest_src) as f:
            manifest = json.load(f)
        if "config" in manifest:
            manifest["config"]["num_experts"] = N
            manifest["config"]["_smoke_test"] = True
            manifest["config"]["_full_num_experts"] = full_num_experts
        with open(manifest_dst, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  patched: model_weights.json (num_experts={N})")
    else:
        print(f"  skip (not found): model_weights.json")

    # Create config.json with num_experts override
    config = {
        "num_experts": N,
        "num_experts_per_tok": 10,  # model default; actual K=4 is hardcoded
        "_smoke_test": True,
        "_source_model": model_dir,
        "_source_experts": args.source_experts,
        "_full_num_experts": full_num_experts,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  wrote: config.json (num_experts={N})")

    # Truncate expert files
    out_expert_dir = os.path.join(out_dir, args.source_experts)
    os.makedirs(out_expert_dir, exist_ok=True)

    for layer_idx in range(args.num_layers):
        src_path = os.path.join(src_expert_dir, f"layer_{layer_idx:02d}.bin")
        dst_path = os.path.join(out_expert_dir, f"layer_{layer_idx:02d}.bin")

        if not os.path.isfile(src_path):
            print(f"  WARNING: {src_path} not found, skipping")
            continue

        # Read only the first N experts
        with open(src_path, "rb") as fin:
            data = fin.read(subset_layer_size)

        if len(data) != subset_layer_size:
            print(f"  WARNING: layer {layer_idx} read {len(data)} bytes, expected {subset_layer_size}")

        with open(dst_path, "wb") as fout:
            fout.write(data)

        pct = (layer_idx + 1) / args.num_layers * 100
        print(f"\r  experts: layer {layer_idx:02d}/{args.num_layers-1} ({pct:.0f}%)", end="", flush=True)

    print()

    # Also symlink layout.json if present
    src_layout = os.path.join(src_expert_dir, "layout.json")
    if os.path.isfile(src_layout):
        # Create modified layout with correct num_experts
        with open(src_layout) as f:
            layout = json.load(f)
        layout["num_experts"] = N
        layout["_smoke_test"] = True
        dst_layout = os.path.join(out_expert_dir, "layout.json")
        with open(dst_layout, "w") as f:
            json.dump(layout, f, indent=2)
        print(f"  wrote: {args.source_experts}/layout.json (num_experts={N})")

    print()
    print(f"Done! Smoke model at: {out_dir}")
    print(f"  Dense weights: symlinked from source")
    print(f"  Experts: {N} per layer × {args.num_layers} layers = {total_size_gb:.2f} GB")
    print()
    print(f"Run with:")
    print(f"  ./infer --model {out_dir} --2bit --prompt \"Hello\" --tokens 20 --stream")

if __name__ == "__main__":
    main()
