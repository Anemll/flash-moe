#!/usr/bin/env python3
"""
Analyze routing log to find top-N hottest expert IDs globally.

The model has 512 experts shared across all 60 layers (same expert ID space).
This finds the N expert IDs that are called most often across all layers combined.

Usage:
    python3 analyze_routing.py /tmp/routing_ane.bin --top 32
"""

import argparse
import struct
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("routing_log", help="Binary routing log from --collect-routing")
    parser.add_argument("--top", type=int, default=32, help="Top N experts to keep")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--json", help="Output JSON file with expert list")
    args = parser.parse_args()

    # Parse routing log: each record = [layer_idx(i32), K(i32), hidden(f32*4096), expert_ids(i32*K)]
    global_freq = Counter()       # expert_id -> total calls across all layers
    per_layer_freq = {}           # layer -> Counter of expert_id -> calls

    with open(args.routing_log, "rb") as f:
        data = f.read()

    pos = 0
    n_records = 0
    while pos < len(data):
        if pos + 8 > len(data):
            break
        layer_idx, K = struct.unpack_from("ii", data, pos)
        pos += 8
        pos += args.hidden_dim * 4  # skip hidden vector
        if pos + K * 4 > len(data):
            break
        expert_ids = struct.unpack_from(f"{K}i", data, pos)
        pos += K * 4

        if layer_idx not in per_layer_freq:
            per_layer_freq[layer_idx] = Counter()
        for eid in expert_ids:
            global_freq[eid] += 1
            per_layer_freq[layer_idx][eid] += 1
        n_records += 1

    total_calls = sum(global_freq.values())
    unique_experts = len(global_freq)
    print(f"Parsed {n_records} routing records")
    print(f"Total expert calls: {total_calls}")
    print(f"Unique expert IDs seen: {unique_experts}/512\n")

    # Global top N
    top_n = global_freq.most_common(args.top)
    top_n_calls = sum(c for _, c in top_n)
    coverage = top_n_calls / total_calls * 100

    print(f"=== Top {args.top} Hottest Experts (global across all layers) ===")
    print(f"{'Rank':>4}  {'Expert':>6}  {'Calls':>6}  {'% of total':>10}  {'Layers active':>13}")
    print("-" * 55)
    for rank, (eid, count) in enumerate(top_n, 1):
        layers_active = sum(1 for l in per_layer_freq if eid in per_layer_freq[l])
        pct = count / total_calls * 100
        print(f"  {rank:>2}   {eid:>5}   {count:>5}    {pct:>5.2f}%     {layers_active:>3}/60")

    print(f"\nTop-{args.top} coverage: {coverage:.1f}% of all expert calls")
    print(f"Hot expert IDs: {sorted(eid for eid, _ in top_n)}")

    # Show what's NOT covered — per layer stats
    hot_set = set(eid for eid, _ in top_n)
    print(f"\n=== Per-Layer Coverage with these {args.top} experts ===")
    print(f"{'Layer':>5}  {'Calls':>6}  {'Hits':>5}  {'Coverage':>8}  {'Misses use expert':>20}")
    print("-" * 65)
    for l in sorted(per_layer_freq.keys()):
        lf = per_layer_freq[l]
        total = sum(lf.values())
        hits = sum(lf[e] for e in hot_set if e in lf)
        cov = hits / total * 100 if total > 0 else 0
        # Top miss
        misses = [(e, c) for e, c in lf.most_common() if e not in hot_set]
        miss_str = f"#{misses[0][0]}({misses[0][1]}x)" if misses else "none"
        print(f"  {l:>3}   {total:>5}   {hits:>4}    {cov:>5.1f}%    {miss_str}")

    if args.json:
        import json
        out = {
            "top_n": args.top,
            "coverage_pct": round(coverage, 1),
            "expert_ids": sorted(eid for eid, _ in top_n),
            "expert_calls": {str(eid): count for eid, count in top_n},
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json}")

if __name__ == "__main__":
    main()
