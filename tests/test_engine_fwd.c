#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "kernels/cpu_rope.h"
#include "kernels/cpu_sdpa.h"
#include "kernels/cpu_silu.h"

#include "test_engine_helpers.h"

/* ===================== kernel unit tests ================================ */

static int test_silu_matches_scalar(void)
{
    float pts[] = {-3.0f,-1.5f,-0.5f,0.0f,0.25f,0.5f,1.0f,2.0f,3.5f,5.0f};
    int failed = 0;
    for (size_t i = 0; i < sizeof(pts)/sizeof(pts[0]); ++i) {
        float x = pts[i];
        float want = x / (1.0f + expf(-x));
        float got = cpu_silu(x);
        float d = fabsf(got - want);
        if (d > 1e-6f * fabsf(want) && d > 1e-7f) {
            printf("FAIL: silu(%g) got=%.8f want=%.8f\n", x, got, want);
            ++failed;
        }
    }
    if (!failed) printf("PASS: silu matches scalar\n");
    return failed;
}

/* Scalar HF rotate_half RoPE reference (out-of-place). */
static void scalar_rope(olmoe_act_t *out, const olmoe_act_t *x,
                        size_t seq, size_t nh, size_t hd, float theta)
{
    size_t h2 = hd / 2;
    for (size_t i = 0; i < seq; ++i)
        for (size_t hh = 0; hh < nh; ++hh) {
            const olmoe_act_t *src = x + i*nh*hd + hh*hd;
            olmoe_act_t *dst = out + i*nh*hd + hh*hd;
            for (size_t k = 0; k < h2; ++k) {
                float inv = 1.0f/powf(theta, (2.0f*(float)k)/(float)hd);
                float a = cosf((float)i*inv), b = sinf((float)i*inv);
                dst[k]      = src[k]*a - src[h2+k]*b;
                dst[h2+k]   = src[h2+k]*a + src[k]*b;
            }
        }
}

static int test_rope_matches_scalar(void)
{
    enum { SEQ=3, NH=2, HD=8 };
    size_t n = SEQ*NH*HD;
    olmoe_act_t q[n], saved[n], ref[n];
    for (size_t i = 0; i < n; ++i)
        q[i] = (float)(((int)(i*7 % 23)) - 11) / 13.0f;
    memcpy(saved, q, sizeof q);

    /* pre-norm per head for norm-preservation check */
    float pre_norm[NH*SEQ];
    for (size_t i = 0; i < SEQ; ++i)
        for (size_t hh = 0; hh < NH; ++hh) {
            float s = 0.0f;
            for (size_t d = 0; d < HD; ++d) { float v = q[i*NH*HD+hh*HD+d]; s += v*v; }
            pre_norm[hh*SEQ+i] = s;
        }

    cpu_rope(q, SEQ, NH, HD, OLMOE_ROPE_THETA);
    scalar_rope(ref, saved, SEQ, NH, HD, OLMOE_ROPE_THETA);

    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(q[i] - ref[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(ref[i])) {
            printf("FAIL: rope lane %zu got=%.7f want=%.7f\n", i, q[i], ref[i]);
            ++failed;
        }
    }
    /* norm preservation: RoPE is a rotation per (head, pair), so per-head
     * sum of squares is invariant. */
    for (size_t i = 0; i < SEQ && !failed; ++i)
        for (size_t hh = 0; hh < NH && !failed; ++hh) {
            float s = 0.0f;
            for (size_t d = 0; d < HD; ++d) { float v = q[i*NH*HD+hh*HD+d]; s += v*v; }
            if (fabsf(s - pre_norm[hh*SEQ+i]) > 1e-3f * pre_norm[hh*SEQ+i]) {
                printf("FAIL: rope norm not preserved head %zu tok %zu\n", hh, i);
                ++failed;
            }
        }
    if (!failed) printf("PASS: rope matches scalar\n");
    return failed;
}

/* Scalar causal SDPA reference (out-of-place). */
static void scalar_sdpa(olmoe_act_t *out, const olmoe_act_t *q,
                        const olmoe_act_t *k, const olmoe_act_t *v,
                        size_t seq, size_t nh, size_t hd, float scale)
{
    size_t st = nh*hd;
    for (size_t hh = 0; hh < nh; ++hh)
        for (size_t i = 0; i < seq; ++i) {
            const olmoe_act_t *qi = q + i*st + hh*hd;
            float scores[256];
            float mx = -INFINITY;
            for (size_t j = 0; j <= i; ++j) {
                float dot = 0.0f;
                const olmoe_act_t *kj = k + j*st + hh*hd;
                for (size_t d = 0; d < hd; ++d) dot += qi[d]*kj[d];
                scores[j] = dot*scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= i; ++j) { scores[j] = expf(scores[j]-mx); sum += scores[j]; }
            olmoe_act_t *oi = out + i*st + hh*hd;
            for (size_t d = 0; d < hd; ++d) oi[d] = 0.0f;
            for (size_t j = 0; j <= i; ++j) {
                float w = scores[j]/sum;
                const olmoe_act_t *vj = v + j*st + hh*hd;
                for (size_t d = 0; d < hd; ++d) oi[d] += w*vj[d];
            }
        }
}

static int test_sdpa_matches_scalar(void)
{
    enum { SEQ=4, NH=2, HD=4 };
    size_t n = SEQ*NH*HD;
    olmoe_act_t q[n], k[n], v[n], got[n], want[n];
    for (size_t i = 0; i < n; ++i) {
        q[i] = (float)(((int)(i*3 % 11)) - 5) / 7.0f;
        k[i] = (float)(((int)(i*5 % 13)) - 6) / 9.0f;
        v[i] = (float)(((int)(i*7 % 17)) - 8) / 11.0f;
    }
    float scale = 1.0f/sqrtf((float)HD);
    cpu_sdpa(got, q, k, v, SEQ, NH, HD, scale);
    scalar_sdpa(want, q, k, v, SEQ, NH, HD, scale);

    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(want[i])) {
            printf("FAIL: sdpa lane %zu got=%.7f want=%.7f\n", i, got[i], want[i]);
            ++failed;
        }
    }
    if (failed) return failed;

    /* causal property: perturb the LAST key (only query seq-1 attends to it)
     * and assert earlier-query outputs are unchanged. */
    olmoe_act_t kp[n], got2[n];
    memcpy(kp, k, sizeof k);
    for (size_t d = 0; d < NH*HD; ++d) kp[(SEQ-1)*NH*HD + d] += 123.456f;
    cpu_sdpa(got2, q, kp, v, SEQ, NH, HD, scale);
    for (size_t i = 0; i < SEQ-1 && !failed; ++i)
        for (size_t j = 0; j < NH*HD && !failed; ++j) {
            size_t p = i*NH*HD + j;
            float d = fabsf(got[p] - got2[p]);
            if (d > 1e-6f) {
                printf("FAIL: sdpa causal leak tok %zu lane %zu d=%g\n", i, j, d);
                ++failed;
            }
        }
    if (!failed) printf("PASS: sdpa matches scalar\n");
    return failed;
}

/* ===================== E2E scalar forward reference ===================== */

static const float LN_EPS = 1e-5f;

static void scalar_rmsnorm(olmoe_act_t *out, const olmoe_act_t *x,
                           const olmoe_bf16_t *w, size_t seq, size_t n)
{
    for (size_t i = 0; i < seq; ++i) {
        const olmoe_act_t *xr = x + i*n;
        float ss = 0.0f;
        for (size_t k = 0; k < n; ++k) ss += xr[k]*xr[k];
        float sc = 1.0f/sqrtf(ss/(float)n + LN_EPS);
        for (size_t k = 0; k < n; ++k)
            out[i*n+k] = xr[k]*sc*bf16_to_f32(w[k]);
    }
}

static void scalar_rmsnorm_inplace(olmoe_act_t *x, const olmoe_bf16_t *w,
                                   size_t seq, size_t n)
{
    for (size_t i = 0; i < seq; ++i) {
        olmoe_act_t *xr = x + i*n;
        float ss = 0.0f;
        for (size_t k = 0; k < n; ++k) ss += xr[k]*xr[k];
        float sc = 1.0f/sqrtf(ss/(float)n + LN_EPS);
        for (size_t k = 0; k < n; ++k) xr[k] = xr[k]*sc*bf16_to_f32(w[k]);
    }
}

/* Per-head RMSNorm for QK norm: normalizes each head's OLMOE_HEAD_DIM lanes
 * independently, using the corresponding slice of the weight vector. */
static void scalar_rmsnorm_qk_inplace(olmoe_act_t *x, const olmoe_bf16_t *w,
                                      size_t seq, size_t n_heads,
                                      size_t head_dim)
{
    size_t n = n_heads * head_dim;
    for (size_t i = 0; i < seq; ++i) {
        for (size_t h = 0; h < n_heads; ++h) {
            size_t off = i * n + h * head_dim;
            olmoe_act_t *xr = x + off;
            float ss = 0.0f;
            for (size_t k = 0; k < head_dim; ++k) ss += xr[k] * xr[k];
            float sc = 1.0f / sqrtf(ss / (float)head_dim + LN_EPS);
            const olmoe_bf16_t *hw = w + h * head_dim;
            for (size_t k = 0; k < head_dim; ++k)
                xr[k] = xr[k] * sc * bf16_to_f32(hw[k]);
        }
    }
}

static void scalar_softmax(float *out, const float *in, size_t n)
{
    float mx = in[0]; for (size_t r=1;r<n;++r) if (in[r]>mx) mx=in[r];
    float s=0.0f; for (size_t r=0;r<n;++r){ out[r]=expf(in[r]-mx); s+=out[r]; }
    for (size_t r=0;r<n;++r) out[r]/=s;
}

static void scalar_topk(const float *sc, size_t n, size_t k, int *idx, float *w)
{
    bool used[64] = {false};
    for (size_t r=0;r<k;++r){
        size_t best=n; float bv=-INFINITY;
        for (size_t j=0;j<n;++j){ if(used[j])continue;
            if(best==n||sc[j]>bv){best=j;bv=sc[j];} }
        used[best]=true; idx[r]=(int)best; w[r]=sc[best];
    }
}

static void scalar_renorm(float *w, size_t k)
{
    float s=0.0f; for (size_t r=0;r<k;++r) s+=w[r];
    for (size_t r=0;r<k;++r) w[r]/=s;
}

static void add_residual(olmoe_act_t *out, const olmoe_act_t *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) out[i] += x[i];
}

/* In-place rotate_half (per-pair temps keep the clobber order correct). */
static void scalar_rope_inplace(olmoe_act_t *x, size_t seq, size_t nh,
                                size_t hd, float theta)
{
    size_t h2 = hd/2;
    for (size_t i = 0; i < seq; ++i)
        for (size_t hh = 0; hh < nh; ++hh) {
            olmoe_act_t *r = x + i*nh*hd + hh*hd;
            for (size_t k = 0; k < h2; ++k) {
                float inv = 1.0f/powf(theta, (2.0f*(float)k)/(float)hd);
                float a = cosf((float)i*inv), b = sinf((float)i*inv);
                float x0 = r[k], x1 = r[h2+k];
                r[k]     = x0*a - x1*b;
                r[h2+k]  = x1*a + x0*b;
            }
        }
}

/* Mirror of olmoe_forward on the same (shared-expert) model. */
static void scalar_forward(const olmoe_model_t *m, const int *ids, size_t seq,
                           olmoe_act_t *logits_out)
{
    size_t H=(size_t)OLMOE_HIDDEN, I=(size_t)OLMOE_INTER, V=(size_t)OLMOE_VOCAB;
    size_t NH=(size_t)OLMOE_NUM_HEADS, HD=(size_t)OLMOE_HEAD_DIM;
    float *h_in=malloc(seq*H*sizeof(float));
    float *h_out=malloc(seq*H*sizeof(float));
    float *normed=malloc(seq*H*sizeof(float));
    float *q=malloc(seq*H*sizeof(float)),*k=malloc(seq*H*sizeof(float));
    float *v=malloc(seq*H*sizeof(float)),*ctx=malloc(seq*H*sizeof(float));
    for (size_t i=0;i<seq;++i)
        for (size_t c=0;c<H;++c)
            h_in[i*H+c]=bf16_to_f32(m->embed_tokens[(size_t)ids[i]*H+c]);
    for (size_t l=0;l<m->n_layers;++l){
        const olmoe_layer_t *L=&m->layers[l];
        scalar_rmsnorm(normed, h_in, L->input_layernorm, seq, H);
        scalar_matmul_bf16(q, normed, L->self_attn.q_proj, seq, H, H);
        scalar_matmul_bf16(k, normed, L->self_attn.k_proj, seq, H, H);
        scalar_matmul_bf16(v, normed, L->self_attn.v_proj, seq, H, H);
        scalar_rmsnorm_qk_inplace(q, L->self_attn.q_norm, seq, NH, HD);
        scalar_rmsnorm_qk_inplace(k, L->self_attn.k_norm, seq, NH, HD);
        scalar_rope_inplace(q, seq, NH, HD, OLMOE_ROPE_THETA);
        scalar_rope_inplace(k, seq, NH, HD, OLMOE_ROPE_THETA);
        scalar_sdpa(ctx, q, k, v, seq, NH, HD, 1.0f/sqrtf((float)HD));
        scalar_matmul_bf16(h_out, ctx, L->self_attn.o_proj, seq, H, H);
        add_residual(h_out, h_in, seq*H);
        scalar_rmsnorm(normed, h_out, L->post_attention_layernorm, seq, H);
        float *rlogits=malloc(seq*64*sizeof(float));
        scalar_matmul_bf16(rlogits, normed, L->mlp_gate, seq, 64, H);
        int *idx=malloc(seq*8*sizeof(int));
        float *w=malloc(seq*8*sizeof(float));
        for (size_t i=0;i<seq;++i){
            float probs[64]; scalar_softmax(probs, rlogits+i*64, 64);
            scalar_topk(probs, 64, 8, idx+i*8, w+i*8);
        }
        float *eout=calloc(seq*H, sizeof(float));
        float *gate=malloc(I*sizeof(float)),*up=malloc(I*sizeof(float));
        float *down=malloc(H*sizeof(float)),*act=malloc(I*sizeof(float));
        for (size_t i=0;i<seq;++i)
            for (size_t r=0;r<8;++r){
                const olmoe_expert_t *e=&L->experts[idx[i*8+r]];
                const olmoe_act_t *tok=normed+i*H;
                scalar_matmul_bf16(gate, tok, e->gate_proj, 1, I, H);
                scalar_matmul_bf16(up, tok, e->up_proj, 1, I, H);
                for (size_t j=0;j<I;++j) act[j]=cpu_silu(gate[j])*up[j];
                scalar_matmul_bf16(down, act, e->down_proj, 1, H, I);
                for (size_t h=0;h<H;++h) eout[i*H+h]+=w[i*8+r]*down[h];
            }
        add_residual(h_out, eout, seq*H);
        memcpy(h_in, h_out, seq*H*sizeof(float));
        free(rlogits);free(idx);free(w);free(eout);
        free(gate);free(up);free(down);free(act);
    }
    scalar_rmsnorm(h_out, h_in, m->norm, seq, H);
    scalar_matmul_bf16(logits_out, h_out, m->lm_head, seq, V, H);
    free(h_in);free(h_out);free(normed);free(q);free(k);free(v);free(ctx);
}

/* ===================== E2E model build / test ========================== */

static void fill_bf16_ramp(olmoe_bf16_t *w, size_t n, unsigned int seed)
{
    unsigned int rng = seed;
    for (size_t i = 0; i < n; ++i) {
        rng = rng*1664525u + 1013904223u;
        int v = (int)(rng % 2001) - 1000;
        w[i] = f32_to_bf16((float)v / 1000.0f);
    }
}

static int test_forward_end_to_end_matches_scalar(void)
{
    enum { SEQ = 2 };
    size_t H=(size_t)OLMOE_HIDDEN, I=(size_t)OLMOE_INTER, V=(size_t)OLMOE_VOCAB;
    olmoe_model_t m; memset(&m, 0, sizeof m);
    m.embed_tokens = malloc(V*H*sizeof(olmoe_bf16_t));
    m.lm_head      = malloc(V*H*sizeof(olmoe_bf16_t));
    m.norm         = malloc(H*sizeof(olmoe_bf16_t));
    m.n_layers = 1;
    m.layers = calloc(1, sizeof(olmoe_layer_t));
    if (!m.embed_tokens || !m.lm_head || !m.norm || !m.layers) {
        printf("FAIL: e2e model malloc OOM\n");
        return 1;
    }
    fill_bf16_ramp(m.embed_tokens, V*H, 0xabc0def1u);
    fill_bf16_ramp(m.lm_head, (size_t)V*H, 0x55aa55aau);
    fill_bf16_ramp(m.norm, H, 0x12345678u);

    olmoe_layer_t *L = &m.layers[0];
    L->self_attn.q_proj = malloc(H*H*sizeof(olmoe_bf16_t));
    L->self_attn.k_proj = malloc(H*H*sizeof(olmoe_bf16_t));
    L->self_attn.v_proj = malloc(H*H*sizeof(olmoe_bf16_t));
    L->self_attn.o_proj = malloc(H*H*sizeof(olmoe_bf16_t));
    L->self_attn.q_norm = malloc(H*sizeof(olmoe_bf16_t));
    L->self_attn.k_norm = malloc(H*sizeof(olmoe_bf16_t));
    L->input_layernorm          = malloc(H*sizeof(olmoe_bf16_t));
    L->post_attention_layernorm = malloc(H*sizeof(olmoe_bf16_t));
    L->mlp_gate = malloc(64*H*sizeof(olmoe_bf16_t));
    /* Shared nonzero expert weights: all 64 experts alias the same three
     * buffers so memory stays at ~16 MiB but the SiLU-gated up/down path
     * is genuinely exercised (the scalar reference shares the same alias). */
    olmoe_bf16_t *sg = malloc(I*H*sizeof(olmoe_bf16_t));
    olmoe_bf16_t *su = malloc(I*H*sizeof(olmoe_bf16_t));
    olmoe_bf16_t *sd = malloc(H*I*sizeof(olmoe_bf16_t));
    fill_bf16_ramp(L->self_attn.q_proj, H*H, 0x0bad1deau);
    fill_bf16_ramp(L->self_attn.k_proj, H*H, 0xfeed1234u);
    fill_bf16_ramp(L->self_attn.v_proj, H*H, 0x9001cafeu);
    fill_bf16_ramp(L->self_attn.o_proj, H*H, 0xbeefb00bu);
    fill_bf16_ramp(L->self_attn.q_norm, H, 0xaaaa0101u);
    fill_bf16_ramp(L->self_attn.k_norm, H, 0xbbbb0202u);
    fill_bf16_ramp(L->input_layernorm, H, 0xcccc0303u);
    fill_bf16_ramp(L->post_attention_layernorm, H, 0xdddd0404u);
    fill_bf16_ramp(L->mlp_gate, 64*H, 0xeeee0505u);
    fill_bf16_ramp(sg, I*H, 0x11111717u);
    fill_bf16_ramp(su, I*H, 0x22222828u);
    fill_bf16_ramp(sd, H*I, 0x33333939u);
    for (int e = 0; e < OLMOE_N_EXPERTS; ++e) {
        L->experts[e].gate_proj = sg;
        L->experts[e].up_proj   = su;
        L->experts[e].down_proj = sd;
    }

    int ids[SEQ] = {5, 7};
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, SEQ);
    olmoe_act_t *got = s.logits;
    olmoe_status_t st = olmoe_forward(&m, ids, SEQ, &s, got);
    if (st != OLMOE_OK) { printf("FAIL: e2e forward -> %d\n", st); return 1; }

    olmoe_act_t *want = malloc(SEQ*V*sizeof(olmoe_act_t));
    scalar_forward(&m, ids, SEQ, want);

    /* Through BF16 weights + FMA-reordered SIMD accumulation, FP32 drift
     * over one layer + the vocab-wide lm_head contraction is larger than a
     * single matmul's rtol 1e-4; rtol 1e-2 / atol 1e-3 is the honest budget. */
    const float RTOL = 1e-2f, ATOL = 1e-3f;
    int failed = 0; float maxd = 0.0f; size_t worst = 0;
    for (size_t i = 0; i < (size_t)SEQ*V; ++i) {
        float d = fabsf(got[i]-want[i]);
        if (d > maxd) { maxd = d; worst = i; }
        if (d > ATOL && d > RTOL*fabsf(want[i])) {
            if (failed < 5)
                printf("FAIL: e2e logits[%zu] got=%.6f want=%.6f\n",
                       i, got[i], want[i]);
            ++failed;
        }
    }
    if (!failed) printf("PASS: forward end-to-end matches scalar "
                        "(maxd=%.4g at lane %zu; tol rtol=%g atol=%g)\n",
                        maxd, worst, RTOL, ATOL);
    else          printf("FAIL: e2e max diff=%.4g at lane %zu\n", maxd, worst);

    olmoe_scratch_free(&s);
    free(want);
    for (int e = 0; e < OLMOE_N_EXPERTS; ++e) { L->experts[e].gate_proj=NULL;
        L->experts[e].up_proj=NULL; L->experts[e].down_proj=NULL; }
    free(sg); free(su); free(sd);
    free(L->self_attn.q_proj);free(L->self_attn.k_proj);
    free(L->self_attn.v_proj);free(L->self_attn.o_proj);
    free(L->self_attn.q_norm);free(L->self_attn.k_norm);
    free(L->input_layernorm);free(L->post_attention_layernorm);
    free(L->mlp_gate); free(m.layers);
    free(m.embed_tokens); free(m.lm_head); free(m.norm);
    return failed;
}

int test_engine_fwd_pass(void)
{
    int failed = 0;
    failed += test_silu_matches_scalar();
    failed += test_rope_matches_scalar();
    failed += test_sdpa_matches_scalar();
    failed += test_forward_end_to_end_matches_scalar();
    return failed;
}