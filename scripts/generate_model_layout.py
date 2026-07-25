#!/usr/bin/env python3
"""Build-time codegen: bake the OLMoE safetensors layout into a C .inc.

The runtime loader (`src/olmoe/layers/layers.c`) is forbidden from parsing
JSON: it only reads `model-*.safetensors` shards, skips the 8-byte LE
header-length + JSON header, then reads tensors sequentially in the baked
order. Everything the loader needs to know about WHICH tensor lives in
WHICH shard and in WHAT read-order is produced here.

Why no runtime JSON: see docs/layer_module.md and AGENTS.md "Integration
& verification". Source of truth for the oracle is this script plus
`scripts/inspect_safetensors.py --dump-values` for cross-checking at dev
time (both use the same bf16_to_f32 helper for the dump).

Output (args: <index_path> <output_inc_path>):
  - OLMOE_* dimension constants
  - olmoe_slot_kind_t enum
  - olmoe_tensor_desc_t struct shape tags
  - OLMOE_SHARD_FILE_NAMES[]  (sorted shard file names)
  - OLMOE_SHARD_LAYOUT[][N]   (per-shard tensor desc arrays, sorted-by-name
                              which equals sorted-by-offset within a shard)
  - OLMOE_SHARD_LEN[]         (per-shard tensor count)
  - OLMOE_N_TOTAL_TENSORS
  - OLMOE_ORACLE_<name>[16]   (raw bf16 uint16 samples for value tests)

Build-time validation: every tensor's bytes-from-dims formula is checked
against the actual safetensors data_offsets span and aborts on mismatch, so
a shape change upstream cannot silently ship a broken layout.
"""
import argparse
import json
import os
import re
import struct
import sys

# Tensor name prefix for oracles is sorted-by-name within a shard; per AGENTS
# "no diagnostic scratch files in the tree" we keep oracle names here.
ORACLE_SAMPLES = [
    ("MODEL_NORM",          "model.norm.weight"),
    ("LM_HEAD",            "lm_head.weight"),
    ("EXPERT0_DOWN",       "model.layers.0.mlp.experts.0.down_proj.weight"),
]
ORACLE_N_VALUES = 16

KIND_ORDER = [
    "EMBED", "LM_HEAD", "NORM",
    "INPUT_LN", "POST_LN",
    "Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ", "Q_NORM", "K_NORM",
    "MLP_GATE",
    "EXPERT_GATE", "EXPERT_UP", "EXPERT_DOWN",
]
KIND_TO_ENUM = {k: "OLMOE_KIND_" + k for k in KIND_ORDER}


def bytes_for_kind(kind, dims):
    """Bytes of one BF16 tensor of this kind, derived from config dims.

    Cross-checked against actual safetensors data_offsets span in main();
    any mismatch raises SystemExit so upstream shape changes can never ship
    a silently-broken layout.
    """
    h, i, v, e = (
        dims["hidden"], dims["inter"], dims["vocab"], dims["n_experts"],
    )
    return {
        "EMBED":        v * h * 2,
        "LM_HEAD":      v * h * 2,
        "NORM":         h * 2,
        "INPUT_LN":     h * 2,
        "POST_LN":      h * 2,
        "Q_PROJ":       h * h * 2,
        "K_PROJ":       h * h * 2,
        "V_PROJ":       h * h * 2,
        "O_PROJ":       h * h * 2,
        "Q_NORM":       h * 2,
        "K_NORM":       h * 2,
        "MLP_GATE":     e * h * 2,
        "EXPERT_GATE":  i * h * 2,
        "EXPERT_UP":    i * h * 2,
        "EXPERT_DOWN":  h * i * 2,
    }[kind]


# Regexes anchored to the OLMoE naming scheme. Layers/expert ints are parsed
# and emitted as compile-time struct tags.
_NAME_PATTERNS = [
    (r"^lm_head\.weight$",                                                              "LM_HEAD",      None, None),
    (r"^model\.embed_tokens\.weight$",                                                 "EMBED",        None, None),
    (r"^model\.norm\.weight$",                                                         "NORM",         None, None),
    (r"^model\.layers\.(?P<L>\d+)\.input_layernorm\.weight$",                          "INPUT_LN",     "L",  None),
    (r"^model\.layers\.(?P<L>\d+)\.post_attention_layernorm\.weight$",                  "POST_LN",      "L",  None),
    (r"^model\.layers\.(?P<L>\d+)\.self_attn\.(?P<k>q|k|v|o)_proj\.weight$",           None,           "L",  "k_proj"),
    (r"^model\.layers\.(?P<L>\d+)\.self_attn\.(?P<k>q|k)_norm\.weight$",               None,           "L",  "k_norm"),
    (r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$",                                 "MLP_GATE",     "L",  None),
    (r"^model\.layers\.(?P<L>\d+)\.mlp\.experts\.(?P<E>\d+)\.(?P<k>gate|up|down)_proj\.weight$", None, "L", "expert_proj"),
]
_KPROJ_KIND = {"q": "Q_PROJ", "k": "K_PROJ", "v": "V_PROJ", "o": "O_PROJ"}
_KNORM_KIND = {"q": "Q_NORM", "k": "K_NORM"}
_EXPERT_KIND = {"gate": "EXPERT_GATE", "up": "EXPERT_UP", "down": "EXPERT_DOWN"}


def classify_tensor(name):
    """Return (kind, layer, expert) for a tensor name, else raise.

    Returns layer=-1 / expert=-1 for top-level tensors. Negatives are the
    sentinel used by the C desc struct (int8_t) for "not applicable".
    """
    for rx, fixed_kind, layer_grp, dyn_kind in _NAME_PATTERNS:
        m = re.match(rx, name)
        if not m:
            continue
        layer = int(m.group(layer_grp)) if layer_grp else -1
        if dyn_kind == "k_proj":
            return _KPROJ_KIND[m.group("k")], layer, -1
        if dyn_kind == "k_norm":
            return _KNORM_KIND[m.group("k")], layer, -1
        if dyn_kind == "expert_proj":
            expert = int(m.group("E"))
            return _EXPERT_KIND[m.group("k")], layer, expert
        # fixed_kind path: for top-level tensors layer_grp is None so layer
        # is already -1; for INPUT_LN/POST_LN/MLP_GATE layer_grp="L" supplied
        # the layer above.
        return fixed_kind, layer, -1
    raise ValueError(f"unrecognized tensor name: {name}")


def read_safetensors_header(path):
    """Return (header_dict, header_size_bytes) for a .safetensors file."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", raw_len)[0]
        header_json = f.read(header_size)
        if len(header_json) < header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    return json.loads(header_json), header_size


def read_bf16_uint16(path, header_size, data_offset, n):
    """Read `n` BF16 values as raw uint16 from a .safetensors shard.

    Mirrors inspect_safetensors.py: data starts at byte 8 + header_size.
    """
    out = []
    with open(path, "rb") as f:
        f.seek(8 + header_size + data_offset)
        raw = f.read(n * 2)
    if len(raw) < n * 2:
        raise ValueError(f"short read for oracle values at {path}")
    for i in range(n):
        out.append(struct.unpack_from("<H", raw, i * 2)[0])
    return out


def emit(out, s):
    out.write(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", help="Path to model.safetensors.index.json")
    ap.add_argument("output", help="Path to emit the .inc file")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.index))
    idx = json.load(open(args.index))
    weight_map = idx["weight_map"]

    cfg_path = os.path.join(base, "config.json")
    cfg = json.load(open(cfg_path))
    dims = {
        "hidden":     cfg["hidden_size"],
        "inter":      cfg["intermediate_size"],
        "vocab":      cfg["vocab_size"],
        "n_experts":  cfg["num_experts"],
        "n_layers":   cfg["num_hidden_layers"],
    }

    shards = sorted(set(weight_map.values()))

    shard_header = {}   # shard -> header dict
    shard_hsize = {}    # shard -> header byte size
    for sh in shards:
        hdr, hsz = read_safetensors_header(os.path.join(base, sh))
        shard_header[sh] = hdr
        shard_hsize[sh] = hsz

    # Per shard: list of (name, info, kind, L, E) sorted by name. Within a
    # safetensors shard, sorted-by-name == sorted-by-offset (verified below),
    # so reading sequentially after the header reproduces the data layout
    # without per-tensor seeks.
    shard_items = {}
    for sh in shards:
        items = []
        for name, info in shard_header[sh].items():
            if name == "__metadata__":
                continue
            kind, L, E = classify_tensor(name)
            items.append((name, info, kind, L, E))
        items.sort(key=lambda t: t[0])
        shard_items[sh] = items

    # Validate: bytes-formula matches actual span, AND there is no inter-tensor
    # padding (contiguous) so sequential reads are exact.
    for sh in shards:
        items = shard_items[sh]
        for name, info, kind, L, E in items:
            actual = info["data_offsets"][1] - info["data_offsets"][0]
            expect = bytes_for_kind(kind, dims)
            if actual != expect:
                raise SystemExit(
                    f"{sh}::{name}: byte-size mismatch "
                    f"formula={expect} actual={actual}"
                )
        for i in range(len(items) - 1):
            gap = items[i + 1][1]["data_offsets"][0] - items[i][1]["data_offsets"][1]
            if gap != 0:
                raise SystemExit(
                    f"{sh}: padding gap of {gap} bytes between "
                    f"{items[i][0]} and {items[i + 1][0]}"
                )

    total_tensors = sum(len(shard_items[sh]) for sh in shards)

    # Oracle samples: read raw BF16 uint16 for the first N values of each.
    oracles = []
    for label, name in ORACLE_SAMPLES:
        sh = weight_map[name]
        info = shard_header[sh][name]
        vals = read_bf16_uint16(
            os.path.join(base, sh),
            shard_hsize[sh], info["data_offsets"][0], ORACLE_N_VALUES,
        )
        oracles.append((label, vals))

    # Emit the .inc.
    with open(args.output, "w") as out:
        emit_header(out, args, dims, shards, total_tensors)
        emit_enum(out)
        emit_struct(out)
        emit_shard_arrays(out, shards, shard_items)
        emit_oracles(out, oracles)

    print(
        f"generate_model_layout.py: wrote {args.output} "
        f"({len(shards)} shards, {total_tensors} tensors)"
    )


def emit_header(out, args, dims, shards, total_tensors):
    emit(out, "/* Auto-generated by scripts/generate_model_layout.py.\n")
    emit(out, " * Source: " + args.index + "\n")
    emit(out, " * Do not edit by hand. Runtime must never read JSON.\n")
    emit(out, " */\n\n")
    emit(out, "#ifndef OLMOE_MODEL_LAYOUT_INC\n")
    emit(out, "#define OLMOE_MODEL_LAYOUT_INC\n\n")
    emit(out, "#include <stdint.h>\n\n")
    emit(out, f"#define OLMOE_N_LAYERS   {dims['n_layers']}\n")
    emit(out, f"#define OLMOE_N_EXPERTS  {dims['n_experts']}\n")
    emit(out, f"#define OLMOE_HIDDEN     {dims['hidden']}\n")
    emit(out, f"#define OLMOE_INTER      {dims['inter']}\n")
    emit(out, f"#define OLMOE_VOCAB      {dims['vocab']}\n")
    emit(out, f"#define OLMOE_N_SHARDS   {len(shards)}\n")
    emit(out, f"#define OLMOE_N_TOTAL_TENSORS {total_tensors}\n\n")
    emit(out, f"#define OLMOE_ORACLE_N_VALUES {ORACLE_N_VALUES}\n\n")


def emit_enum(out):
    emit(out, "typedef enum {\n")
    for i, k in enumerate(KIND_ORDER):
        suffix = "," if i < len(KIND_ORDER) - 1 else ""
        emit(out, f"    {KIND_TO_ENUM[k]} = {i}{suffix}\n")
    emit(out, "} olmoe_slot_kind_t;\n\n")


def emit_struct(out):
    emit(out, "/* Per-tensor positional tag. layer=-1 / expert=-1 mark\n")
    emit(out, " * tensors that are not layer-scoped resp. not expert-scoped.\n")
    emit(out, " */\n")
    emit(out, "typedef struct {\n")
    emit(out, "    int8_t kind;\n")
    emit(out, "    int8_t layer;\n")
    emit(out, "    int8_t expert;\n")
    emit(out, "} olmoe_tensor_desc_t;\n\n")


def emit_shard_arrays(out, shards, shard_items):
    # File names in sorted order; the loader walks them top-to-bottom.
    emit(out, "static const char *const OLMOE_SHARD_FILE_NAMES[OLMOE_N_SHARDS] = {\n")
    for sh in shards:
        emit(out, f'    "{sh}",\n')
    emit(out, "};\n\n")

    emit(out, "static const size_t OLMOE_SHARD_LEN[OLMOE_N_SHARDS] = {\n")
    for sh in shards:
        emit(out, f"    {len(shard_items[sh])},\n")
    emit(out, "};\n\n")

    for si, sh in enumerate(shards):
        items = shard_items[sh]
        emit(out, f"/* Shard {si}: {sh} ({len(items)} tensors) */\n")
        emit(out, f"static const olmoe_tensor_desc_t "
                   f"OLMOE_SHARD_LAYOUT_{si}[{len(items)}] = {{\n")
        for name, info, kind, L, E in items:
            ek = KIND_TO_ENUM[kind]
            layer_c = f"+{L}" if L >= 0 else "-1"
            expert_c = f"+{E}" if E >= 0 else "-1"
            emit(out, f"    {{ {ek}, {layer_c}, {expert_c} }},"
                       f"  /* {name} */\n")
        emit(out, "};\n\n")

    # Pointer-of-array accessor: layers.c references OLMOE_SHARD_LAYOUT[i]
    # via these so the per-shard arrays share one index name.
    emit(out, "static const olmoe_tensor_desc_t *const\n")
    emit(out, "    OLMOE_SHARD_LAYOUT[OLMOE_N_SHARDS] = {\n")
    for si in range(len(shards)):
        emit(out, f"    OLMOE_SHARD_LAYOUT_{si},\n")
    emit(out, "};\n\n")


def emit_oracles(out, oracles):
    emit(out, "/* Raw BF16 uint16 oracle samples for the first\n")
    emit(out, f" * {ORACLE_N_VALUES} values of selected tensors. Tests memcmp\n")
    emit(out, " * these against the loaded model's pointers. Cross-checked\n")
    emit(out, " * during dev with `scripts/inspect_safetensors.py --dump-values`. */\n")
    for label, vals in oracles:
        emit(out, f"static const uint16_t "
                   f"OLMOE_ORACLE_{label}[OLMOE_ORACLE_N_VALUES] = {{\n    ")
        for i, v in enumerate(vals):
            emit(out, f"0x{v:04x}")
            if i < len(vals) - 1:
                emit(out, ", " if (i + 1) % 8 else ",\n    ")
        emit(out, "\n};\n\n")
    emit(out, "#endif /* OLMOE_MODEL_LAYOUT_INC */\n")


if __name__ == "__main__":
    main()