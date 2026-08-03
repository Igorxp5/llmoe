/* OLMoE model loader: read model-*.safetensors shards directly into the
 * typed hierarchical `olmoe_model_t` without ever parsing JSON at runtime.
 *
 * The full tensor layout (which shard, which slot, which byte-count per
 * kind) is baked by scripts/generate_model_layout.py and embedded here via
 * layers.h. At runtime this file just:
 *   1. allocates one `olmoe_model_t` via calloc (~12.9 GiB, zero-init'd),
 *   2. opens each shard listed in OLMOE_SHARD_FILE_NAMES (in order),
 *   3. skips the 8-byte LE header-length + JSON header,
 *   4. walks OLMOE_SHARD_LAYOUT[shard] and fread()s `bytes_for_kind(kind)`
 *      bytes directly into the array-field at the right offset,
 *   5. returns the populated struct (or NULL + free on any error).
 *
 * Rationale (no runtime JSON): see docs/layer_module.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "olmoe/layers/layers.h"

/* Bytes of one BF16 tensor of `kind`, derived from the baked dim
 * constants. Kept here (not in layer.h) because it is loader-internal.
 * The generator cross-checks every tensor's actual safetensors span against
 * this formula at build time, so a shape drift upstream aborts the build. */
static size_t bytes_for_kind(olmoe_slot_kind_t kind)
{
    switch (kind) {
    case OLMOE_KIND_EMBED:
    case OLMOE_KIND_LM_HEAD:
        return (size_t)OLMOE_VOCAB * OLMOE_HIDDEN * 2;
    case OLMOE_KIND_NORM:
    case OLMOE_KIND_INPUT_LN:
    case OLMOE_KIND_POST_LN:
    case OLMOE_KIND_Q_NORM:
    case OLMOE_KIND_K_NORM:
        return (size_t)OLMOE_HIDDEN * 2;
    case OLMOE_KIND_Q_PROJ:
    case OLMOE_KIND_K_PROJ:
    case OLMOE_KIND_V_PROJ:
    case OLMOE_KIND_O_PROJ:
        return (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN * 2;
    case OLMOE_KIND_MLP_GATE:
        return (size_t)OLMOE_N_EXPERTS * OLMOE_HIDDEN * 2;
    case OLMOE_KIND_EXPERT_GATE:
    case OLMOE_KIND_EXPERT_UP:
        return (size_t)OLMOE_INTER * OLMOE_HIDDEN * 2;
    case OLMOE_KIND_EXPERT_DOWN:
        return (size_t)OLMOE_HIDDEN * OLMOE_INTER * 2;
    }
    return 0;
}

/* Return a pointer to the array field inside `m` identified by `d`.
 * The caller fread's `bytes_for_kind(d->kind)` bytes into this pointer. */
static olmoe_bf16_t *target_field(olmoe_model_t *m,
                                  const olmoe_tensor_desc_t *d)
{
    switch (d->kind) {
    case OLMOE_KIND_EMBED:    return m->embed_tokens;
    case OLMOE_KIND_LM_HEAD:  return m->lm_head;
    case OLMOE_KIND_NORM:     return m->norm;
    case OLMOE_KIND_INPUT_LN:
        return m->layers[d->layer].input_layernorm;
    case OLMOE_KIND_POST_LN:
        return m->layers[d->layer].post_attention_layernorm;
    case OLMOE_KIND_Q_PROJ:
        return m->layers[d->layer].self_attn.q_proj;
    case OLMOE_KIND_K_PROJ:
        return m->layers[d->layer].self_attn.k_proj;
    case OLMOE_KIND_V_PROJ:
        return m->layers[d->layer].self_attn.v_proj;
    case OLMOE_KIND_O_PROJ:
        return m->layers[d->layer].self_attn.o_proj;
    case OLMOE_KIND_Q_NORM:
        return m->layers[d->layer].self_attn.q_norm;
    case OLMOE_KIND_K_NORM:
        return m->layers[d->layer].self_attn.k_norm;
    case OLMOE_KIND_MLP_GATE:
        return m->layers[d->layer].mlp_gate;
    case OLMOE_KIND_EXPERT_GATE:
        return m->layers[d->layer].experts[d->expert].gate_proj;
    case OLMOE_KIND_EXPERT_UP:
        return m->layers[d->layer].experts[d->expert].up_proj;
    case OLMOE_KIND_EXPERT_DOWN:
        return m->layers[d->layer].experts[d->expert].down_proj;
    }
    return NULL;
}

static int skip_header(FILE *f)
{
    unsigned char hdr_len_buf[8];
    if (fread(hdr_len_buf, 1, 8, f) != 8)
        return 0;
    unsigned long header_size = 0;
    for (int i = 0; i < 8; ++i)
        header_size |= (unsigned long)hdr_len_buf[i] << (8 * i);
    if (fseek(f, (long)header_size, SEEK_CUR) != 0)
        return 0;
    return 1;
}

static char *join_shard_path(char *out, size_t cap,
                             const char *dir, const char *shard)
{
    size_t dir_len = strlen(dir);
    if (dir_len + strlen(shard) + 2 > cap)
        return NULL;
    memcpy(out, dir, dir_len);
    if (dir_len > 0 && out[dir_len - 1] != '/')
        out[dir_len++] = '/';
    strcpy(out + dir_len, shard);
    return out;
}

/* Read one shard top-to-bottom, directly into the static model's fields.
 * Returns 0 on failure; the caller frees the single allocation. */
static int load_shard(olmoe_model_t *m, const char *dir,
                      size_t shard_idx)
{
    char path[4096];
    if (!join_shard_path(path, sizeof path, dir,
                        OLMOE_SHARD_FILE_NAMES[shard_idx])) {
        fprintf(stderr, "olmoe_model_load: path too long\n");
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "olmoe_model_load: cannot open %s\n", path);
        return 0;
    }
    if (!skip_header(f)) {
        fprintf(stderr, "olmoe_model_load: bad header in %s\n", path);
        fclose(f);
        return 0;
    }

    const olmoe_tensor_desc_t *layout = OLMOE_SHARD_LAYOUT[shard_idx];
    size_t n = OLMOE_SHARD_LEN[shard_idx];
    for (size_t i = 0; i < n; ++i) {
        const olmoe_tensor_desc_t *d = &layout[i];
        size_t bytes = bytes_for_kind(d->kind);
        olmoe_bf16_t *dest = target_field(m, d);
        if (!dest) {
            fprintf(stderr,
                "olmoe_model_load: unrecognized kind %d\n", d->kind);
            fclose(f);
            return 0;
        }
        if (fread(dest, 1, bytes, f) != bytes) {
            fprintf(stderr,
                "olmoe_model_load: short read in %s at tensor %zu\n",
                path, i);
            fclose(f);
            return 0;
        }
    }

    int extra = fgetc(f);
    if (extra != EOF) {
        fprintf(stderr,
            "olmoe_model_load: trailing bytes in %s (shard drifted "
            "from baked layout)\n", path);
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

olmoe_model_t *olmoe_model_load(const char *dir)
{
    if (!dir) {
        fprintf(stderr, "olmoe_model_load: NULL dir\n");
        return NULL;
    }

    olmoe_model_t *m = calloc(1, sizeof *m);
    if (!m) {
        fprintf(stderr, "olmoe_model_load: calloc(%zu) failed\n",
                sizeof *m);
        return NULL;
    }

    /* Hint for 2 MiB huge pages (THP). Must be set before the shard fread's
     * fault the pages in, so the kernel allocates huge pages on first touch
     * instead of collapsing 4 KiB pages later. MoE decode gathers ~24K
     * scattered 4 KiB expert pages per token; 2 MiB pages collapse that to
     * ~48 TLB entries. Non-fatal: on a kernel without THP the mapping just
     * stays on 4 KiB pages. */
    if (madvise(m, sizeof *m, MADV_HUGEPAGE) != 0)
        perror("olmoe_model_load: madvise(MADV_HUGEPAGE) failed");

    for (size_t s = 0; s < OLMOE_N_SHARDS; ++s) {
        if (!load_shard(m, dir, s)) {
            free(m);
            return NULL;
        }
    }

    if (mlock(m, sizeof *m) != 0)
        perror("olmoe_model_load: mlock failed, weights may be swappable");

    return m;
}

void olmoe_model_free(olmoe_model_t *model)
{
    if (model) {
        munlock(model, sizeof *model);
        free(model);
    }
}
