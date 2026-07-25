#!/usr/bin/env python3
"""Read safetensors index and shard headers, output tensor metadata."""
import argparse
import json
import os
import struct
import sys


def format_size(n_bytes):
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes:,} ({n_bytes / (1024**3):.2f} GiB)"
    elif n_bytes >= 1024 ** 2:
        return f"{n_bytes:,} ({n_bytes / (1024**2):.2f} MiB)"
    elif n_bytes >= 1024:
        return f"{n_bytes:,} ({n_bytes / 1024:.2f} KiB)"
    return f"{n_bytes:,} bytes"


def read_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"File too small to contain safetensors header: {filepath}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)
        if len(header_json) < header_size:
            raise ValueError(f"Truncated file, could not read full header: {filepath}")
    try:
        return json.loads(header_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in safetensors header: {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect safetensors shards via index — print tensor metadata"
    )
    parser.add_argument(
        "index", help="Path to model.safetensors.index.json"
    )
    parser.add_argument(
        "--tensors", "-t", nargs="+",
        help="Filter by tensor name (exact match or prefix)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="0-indexed start position in sorted tensor list (inclusive, default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="0-indexed end position in sorted tensor list (exclusive, default: all)"
    )
    args = parser.parse_args()

    with open(args.index) as f:
        try:
            index = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in index file: {e}", file=sys.stderr)
            sys.exit(1)

    metadata = index.get("metadata", {})
    weight_map = index.get("weight_map", {})

    print(f"Index:  {args.index}")
    print(f"Total size: {format_size(metadata.get('total_size', 0))}")
    print(f"Total tensors: {len(weight_map)}")

    if args.tensors:
        prefixes = set(args.tensors)
        matched = {}
        for name, shard in weight_map.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    matched[name] = shard
        if not matched:
            print(f"\nError: no tensors matched filters: {args.tensors}", file=sys.stderr)
            sys.exit(1)
        if len(matched) < len(weight_map):
            print(f"Matched: {len(matched)} tensor(s)")
        tensors_to_show = matched
    else:
        tensors_to_show = weight_map

    sorted_names = sorted(tensors_to_show)
    if args.end is not None or args.start > 0:
        start = args.start
        end = args.end if args.end is not None else len(sorted_names)
        sliced = sorted_names[start:end]
        tensors_to_show = {name: tensors_to_show[name] for name in sliced}
        print(f"Range:  [{start}:{end}]  ({len(sorted_names)} total)")

    shards_needed = set(tensors_to_show.values())
    base_dir = os.path.dirname(os.path.abspath(args.index))

    shard_headers = {}
    for shard_file in sorted(shards_needed):
        path = os.path.join(base_dir, shard_file)
        try:
            shard_headers[shard_file] = read_safetensors_header(path)
        except (ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    total_data = 0
    missing = 0

    for name in sorted(tensors_to_show):
        shard = tensors_to_show[name]
        header = shard_headers[shard]
        info = header.get(name)
        if info is None:
            missing += 1
            print(f"\nWarning: '{name}' not found in shard '{shard}'", file=sys.stderr)
            continue

        dtype = info.get("dtype", "?")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])
        n_bytes = offsets[1] - offsets[0]
        total_data += n_bytes

        print(f"\n  {name}")
        print(f"    shard:  {shard}")
        print(f"    dtype:  {dtype}")
        print(f"    shape:  {shape}")
        print(f"    bytes:  {format_size(n_bytes)}  (offset {offsets[0]:,} → {offsets[1]:,})")

    print()
    print(f"Shards read: {len(shard_headers)}")
    print(f"Tensors shown: {len(tensors_to_show) - missing}")
    if missing:
        print(f"Tensors missing from shard: {missing}")
    print(f"Total data: {format_size(total_data)}")


if __name__ == "__main__":
    main()
