/* OLMoE model loader: read model-*.safetensors shards into the typed
 * hierarchical `olmoe_model_t` without ever parsing JSON at runtime.
 *
 * The full tensor layout (which shard, which slot, which byte-count per
 * kind) is baked by scripts/generate_model_layout.py and embedded here via
 * layer.h. At runtime this file just:
 *   1. opens each shard listed in OLMOE_SHARD_FILE_NAMES (in order),
 *   2. skips the 8-byte LE header-length + JSON header,
 *   3. walks OLMOE_SHARD_LAYOUT[shard] and fread()s `bytes_for_kind(kind)`
 *      bytes into a freshly malloc'd buffer,
 *   4. routes the buffer to the right field of the model struct.
 *
 * Rationale (no runtime JSON): see docs/layer_module.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Unreachable for a well-formed layout, but C does not know the enum
     * is exhaustive. Avoid -Wreturn-type by returning 0; placement will
     * then malloc(0) and fail the fread guard. */
    return 0;
}

/* Route a freshly-read buffer into the model struct. Also records it in the
 * flat ownership table at `slot`, so cleanup never needs to walk the
 * hierarchy. */
static int place_tensor(olmoe_model_t *m,
                        const olmoe_tensor_desc_t *d,
                        olmoe_bf16_t *buf, size_t slot)
{
    m->bufs[slot] = buf;
    switch (d->kind) {
    case OLMOE_KIND_EMBED:    m->embed_tokens = buf; break;
    case OLMOE_KIND_LM_HEAD:  m->lm_head = buf; break;
    case OLMOE_KIND_NORM:     m->norm = buf; break;
    case OLMOE_KIND_INPUT_LN:
        m->layers[d->layer].input_layernorm = buf; break;
    case OLMOE_KIND_POST_LN:
        m->layers[d->layer].post_attention_layernorm = buf; break;
    case OLMOE_KIND_Q_PROJ:
        m->layers[d->layer].self_attn.q_proj = buf; break;
    case OLMOE_KIND_K_PROJ:
        m->layers[d->layer].self_attn.k_proj = buf; break;
    case OLMOE_KIND_V_PROJ:
        m->layers[d->layer].self_attn.v_proj = buf; break;
    case OLMOE_KIND_O_PROJ:
        m->layers[d->layer].self_attn.o_proj = buf; break;
    case OLMOE_KIND_Q_NORM:
        m->layers[d->layer].self_attn.q_norm = buf; break;
    case OLMOE_KIND_K_NORM:
        m->layers[d->layer].self_attn.k_norm = buf; break;
    case OLMOE_KIND_MLP_GATE:
        m->layers[d->layer].mlp_gate = buf; break;
    case OLMOE_KIND_EXPERT_GATE:
        m->layers[d->layer].experts[d->expert].gate_proj = buf; break;
    case OLMOE_KIND_EXPERT_UP:
        m->layers[d->layer].experts[d->expert].up_proj = buf; break;
    case OLMOE_KIND_EXPERT_DOWN:
        m->layers[d->layer].experts[d->expert].down_proj = buf; break;
    }
    return 1;
}

static int skip_header(FILE *f)
{
    /* safetensors layout: [8-byte LE header-length][JSON header][data].
     * We read the 8-byte length, then seek past the JSON header. */
    unsigned char hdr_len_buf[8];
    if (fread(hdr_len_buf, 1, 8, f) != 8)
        return 0;
    /* header_size is u64 LE; 4 GiB headers are impossible in practice. */
    unsigned long header_size = 0;
    for (int i = 0; i < 8; ++i)
        header_size |= (unsigned long)hdr_len_buf[i] << (8 * i);
    if (fseek(f, (long)header_size, SEEK_CUR) != 0)
        return 0;
    return 1;
}

/* Build "dir/OLMOE_SHARD_FILE_NAMES[shard_idx]" into `out`. Returns out on
 * success, NULL if the path would overflow the buffer. */
static char *join_shard_path(char *out, size_t cap,
                             const char *dir, const char *shard)
{
    size_t dir_len = strlen(dir);
    /* +1 for '/', +1 for '\0', so cap must exceed dir_len + shard_len + 2 */
    if (dir_len + strlen(shard) + 2 > cap)
        return NULL;
    memcpy(out, dir, dir_len);
    if (dir_len > 0 && out[dir_len - 1] != '/')
        out[dir_len++] = '/';
    strcpy(out + dir_len, shard);
    return out;
}

/* Read one shard top-to-bottom. Updates `*slot` (the running global tensor
 * index) for the flat ownership table. Returns 0 on failure; the caller
 * then frees whatever has been loaded so far. */
static int load_shard(olmoe_model_t *m, const char *dir,
                      size_t shard_idx, size_t *slot)
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
        olmoe_bf16_t *buf = malloc(bytes ? bytes : 1);
        if (!buf) {
            fprintf(stderr, "olmoe_model_load: oom in %s\n", path);
            fclose(f);
            return 0;
        }
        if (fread(buf, 1, bytes, f) != bytes) {
            fprintf(stderr,
                "olmoe_model_load: short read in %s at tensor %zu\n",
                path, i);
            free(buf);
            fclose(f);
            return 0;
        }
        if (!place_tensor(m, d, buf, *slot)) {
            free(buf);
            fclose(f);
            return 0;
        }
        ++*slot;
    }

    /* The baked layout is the exact data span of the shard; if there are
     * bytes left after the last tensor, the shard drifted from the layout
     * we compiled against. */
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
    if (!m)
        return NULL;
    m->n_layers = OLMOE_N_LAYERS;
    m->layers = calloc(OLMOE_N_LAYERS, sizeof *m->layers);
    if (!m->layers) {
        free(m);
        return NULL;
    }
    m->n_bufs = OLMOE_N_TOTAL_TENSORS;
    m->bufs = calloc(m->n_bufs, sizeof *m->bufs);
    if (!m->bufs) {
        free(m->layers);
        free(m);
        return NULL;
    }

    size_t slot = 0;
    for (size_t s = 0; s < OLMOE_N_SHARDS; ++s) {
        if (!load_shard(m, dir, s, &slot)) {
            olmoe_model_free(m);
            return NULL;
        }
    }
    return m;
}

void olmoe_model_free(olmoe_model_t *model)
{
    if (!model)
        return;
    if (model->bufs) {
        for (size_t i = 0; i < model->n_bufs; ++i)
            free(model->bufs[i]);
        free(model->bufs);
    }
    free(model->layers);
    free(model);
}
