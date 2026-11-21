#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_lfm_base::llm_build_lfm_base(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_build_lfm_base::build_lfm_shortconv(const llama_memory_recurrent_context * mctx_recr,
                                                      llm_graph_input_rs * inp_recr,
                                                      ggml_tensor * cur,
                                                      int il) {
    const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t  n_seqs       = ubatch.n_seqs;
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;
    const int64_t  n_embd = hparams.n_embd;

    auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);
    // {3*n_embd, n_tokens} => {3*n_embd, n_seq_tokens, n_seqs}
    bcx = ggml_reshape_3d(ctx0, bcx, bcx->ne[0], n_seq_tokens, n_seqs);

    constexpr auto n_chunks = 3;
    GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
    const auto chunk_size = bcx->ne[0] / n_chunks;
    auto *     b          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         0 * chunk_size * ggml_element_size(bcx));
    auto *     c          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         1 * chunk_size * ggml_element_size(bcx));
    auto *     x          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         2 * chunk_size * ggml_element_size(bcx));

    auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

    // read conv state
    auto * conv_state = mctx_recr->get_r_l(il);
    auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
    auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, n_embd, n_seqs);

    // causal prepends the state, non-causal pads symmetrically for a centered window
    if (hparams.causal_attn) {
        bx = ggml_concat(ctx0, conv, bx, 0);
    } else {
        const int64_t pad = (hparams.n_shortconv_l_cache - 1) / 2;
        auto * left = ggml_cont(ctx0,
            ggml_view_3d(ctx0, conv, pad, n_embd, n_seqs, conv->nb[1], conv->nb[2], (d_conv - pad) * conv->nb[0]));
        bx = ggml_pad_ext(ctx0, ggml_concat(ctx0, left, bx, 0), 0, pad, 0, 0, 0, 0, 0, 0);
    }
    GGML_ASSERT(bx->ne[0] > conv->ne[0]);

    build_conv_state_store(inp_recr, conv_state, bx, d_conv, 0);

    auto * conv_kernel = model.layers[il].shortconv.conv;
    auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    auto * y = ggml_mul(ctx0, c, conv_out);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);
    y = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);

    return y;
}

ggml_tensor * llm_build_lfm_base::build_lfm_shortconv(ggml_tensor * cur,
                                                      int il) {
    const int64_t d_conv = hparams.n_shortconv_l_cache - 1;

    auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);

    const int64_t chunk_size = bcx->ne[0] / 3;
    auto * b = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 0 * chunk_size * ggml_element_size(bcx));
    auto * c = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 1 * chunk_size * ggml_element_size(bcx));
    auto * x = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 2 * chunk_size * ggml_element_size(bcx));

    // causal from a zero state
    auto * bx = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul(ctx0, b, x)));
    bx = ggml_pad_ext(ctx0, bx, d_conv, 0, 0, 0, 0, 0, 0, 0);

    auto * conv_out = ggml_ssm_conv(ctx0, bx, model.layers[il].shortconv.conv);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    auto * y = ggml_mul(ctx0, c, ggml_reshape_2d(ctx0, conv_out, chunk_size, n_tokens));
    y = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);

    return y;
}

// Cascade FFN (opt-in): LLAMA_CASCADE="c,k" replaces the dense FFN on single-token steps.
// gate matmul stays stock; the two custom ops do exact top-k selection and the gathered projections.
ggml_tensor * llm_build_lfm_base::build_lfm_cascade_ffn(ggml_tensor * cur, int il, int c, int k) {
    struct ggml_tensor * g;
    {
        const char * e = getenv("LLAMA_CASCADE_SKIP");
        if (e != nullptr && strstr(e, "gate") != nullptr) {
            // timing ablation: skip the gate GEMV (the score is then garbage)
            g = ggml_repeat(ctx0, ggml_view_1d(ctx0, cur, 1, 0),
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.layers[il].ffn_gate->ne[1], cur->ne[1]));
        } else {
            g = build_lora_mm(model.layers[il].ffn_gate, cur);
        }
    }
    cb(g, "model.layers.{}.ffn_gate", il);

    struct ggml_tensor * wu  = model.get_cascade_up_w(il);
    struct ggml_tensor * wus = model.get_cascade_up_s(il);
    struct ggml_tensor * sel = ggml_cascade_up(ctx0, g, cur, model.layers[il].ffn_up, wu, wus, c, k);
    cb(sel, "model.layers.{}.ffn_cascade_up", il);

    struct ggml_tensor * w_t = nullptr;
    struct ggml_tensor * s_t = nullptr;
    model.get_cascade_repack(il, &w_t, &s_t);
    // Metal splits the selected list across GGML_CASCADE_DOWN_NTG threadgroups, each producing a
    // full [n_embd] partial that a tiny second dispatch sums (deterministic, no atomics)
    // one row per slice, plus one row whose first word is the atomic counter of the fused down op
    struct ggml_tensor * part = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32,
                                                   (GGML_CASCADE_DOWN_SLICES_MAX + 1) * model.hparams.n_embd);
    struct ggml_tensor * y = ggml_cascade_down(ctx0, sel, model.layers[il].ffn_down, w_t, s_t, part, k, c);
    cb(y, "model.layers.{}.ffn_cascade_down", il);
    {   // LLAMA_CASCADE_EXTRA_NODES: add N cheap ops per layer to price one graph node
        static const int extra = getenv("LLAMA_CASCADE_EXTRA_NODES") ? atoi(getenv("LLAMA_CASCADE_EXTRA_NODES")) : 0;
        for (int r = 0; r < extra; ++r) {
            y = ggml_scale(ctx0, y, 1.0f);
            cb(y, "model.layers.{}.cascade_probe", il);
        }
    }
    return y;
}

static bool lfm_cascade_cfg(int & c, int & k) {
    static int sc = -1, sk = -1;
    if (sc < 0) {
        sc = 0; sk = 0;
        const char * e = getenv("LLAMA_CASCADE");
        if (e != nullptr) {
            int cc = 0, kk = 0;
            if (sscanf(e, "%d,%d", &cc, &kk) == 2 && kk > 0 && cc >= kk) {
                sc = cc; sk = kk;
            } else {
                fprintf(stderr, "%s: ignoring LLAMA_CASCADE=%s (want \"c,k\" with c >= k > 0)\n", __func__, e);
            }
        }
    }
    c = sc; k = sk;
    return sc > 0;
}

ggml_tensor * llm_build_lfm_base::build_lfm_dense_ffn(ggml_tensor * cur,
                                                      int il) {
    {   // timing ablation: remove the whole FFN from the graph (wrong output, speed only)
        const char * e = getenv("LLAMA_CASCADE_SKIP");
        if (e != nullptr && strstr(e, "ffn") != nullptr) {
            return cur;
        }
        // ... or only the first n layers, for a linearity check on the ablation deltas
        static const int skip_layers = getenv("LLAMA_CASCADE_SKIP_FFN_LAYERS")
                                     ? atoi(getenv("LLAMA_CASCADE_SKIP_FFN_LAYERS")) : 0;
        if (il < skip_layers) {
            return cur;
        }
    }
    int cascade_c = 0, cascade_k = 0;
    if (cur->ne[1] == 1 && cur->ne[2] == 1 && cur->ne[3] == 1 && lfm_cascade_cfg(cascade_c, cascade_k)) {
        return build_lfm_cascade_ffn(cur, il, cascade_c, cascade_k);
    }
    GGML_ASSERT(!model.layers[il].ffn_up_b);
    GGML_ASSERT(!model.layers[il].ffn_gate_b);
    GGML_ASSERT(!model.layers[il].ffn_down_b);
    return build_ffn(cur,
        model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
}

ggml_tensor * llm_build_lfm_base::build_lfm_moe_ffn(ggml_tensor * cur,
                                                    int il) {
    return build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            model.layers[il].ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
            il,
            nullptr,
            nullptr,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
}
