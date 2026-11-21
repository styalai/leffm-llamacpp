# LFM3 cascade FFN and vocabulary cut (this fork)

This is a research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), branched at `7fce543`.
It adds one thing: an LFM3 (`lfm3`) FFN that computes fewer units per token, plus a matching cut of the
tied input embedding and output head. Everything else is upstream llama.cpp, unmodified.

The upstream README follows below; this file describes only the fork. Nothing here is proposed for
upstream.

## What it computes

A gated FFN does `silu(x·W_gateᵀ) ⊙ (x·W_upᵀ) · W_down`. The gate has to see every unit — it is the
router — but the up and down projections do not:

1. **score** all `F` units by `|silu(g)|`, keep the top `c` exactly;
2. run the **up** projection for those `c` units only;
3. rescore by the true activation `|silu(g)·u|`, keep the top `k` exactly;
4. run the **down** projection for those `k` units only.

Cost per token becomes `d·(F + c + k)` instead of `3·d·F`. At `c = 5,120`, `k = 2,304` of `F = 7,680`
that is **65.6% of the dense FFN**.

The vocabulary cut trims the tied embedding and head to an id set derived from calibration data. With
65,536 of 128,000 ids the head is **52.4% of the original**, and the dropout is visible in
`logits` only through `CASCADE_EXPAND`, which scatters the trimmed logits back into the full vocabulary
with a `-inf` sentinel, so sampling code is untouched.

Together these are the **-30.0% FLOPs/token** point for LFM3-1.7B, with -7.4% parameters.

Selection is exact, not thresholded. CUDA and Metal use a four-pass radix select on the float bit
pattern (positive floats are monotone in their bits); thresholds cannot hold a per-token count and
were measured to prune far less than they claim.

## Measured

`llama-bench -p 0 -n 32 -r 3 -ngl 999 --no-warmup`, Q8_0 weights, no quantization of the gate or the
gathered matrices.

| arm | L4 (CUDA) | M5 (Metal) |
| --- | --- | --- |
| dense | 9.014 ms/token | 15.16 ms/token |
| cascade `c=5120 k=2304` | 8.085 ms (1.09x) | 13.32 ms (1.14x) |
| cascade + vocabulary cut | **7.537 ms (1.19x)** | **12.47 ms (1.22x)** |

The gap between "-30% FLOPs" and "1.19x" is the interesting part: at batch 1 this decode is
bandwidth-bound, so cutting arithmetic buys less than it looks like it should, and the two selections
cost ~11.5 µs of dispatch each. Per-stage budgets and the whole tuning ledger are in the fork's commit
messages.

Quality, measured separately on one 2.67M-token continued-pretraining run (one corpus, perplexity
only): after matched training the cut + cascade is **+6.8% bits/byte** against a dense model trained
for the same number of steps, and +0.082 nats zero-shot. Treat that as the price of the recipe on that
corpus, not as a general statement.

At `LLAMA_CASCADE=7680,7680` the cascade reproduces dense greedy output exactly, which is the check
that everything below the selection is arithmetically the dense path.

## Running it

Requires an LFM3 model whose FFN and head match the ids you feed it, and Q8_0 up/down weights.

```sh
# CUDA (sm_89 shown; FlashAttention off keeps the build short and changes nothing at 32 tokens)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_FA=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

# Metal: the default build on macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-cli

# dense baseline, then the cascade, then the cascade with the id cut
./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "..." -n 32 --no-repack
LLAMA_CASCADE=5120,2304 ./build/bin/llama-cli -m ... --no-repack
LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=trim_ids.txt ./build/bin/llama-cli -m ... --no-repack
```

`LLAMA_CASCADE_VOCAB` is a text file of kept ids, one per line, ascending. The same list must have been
used to build the model's trimmed head and embedding, otherwise the ids point at the wrong rows.

The cascade applies to single-token steps only (`T = 1`). Prompt processing stays dense.

## Diagnostics

Every knob is off by default; none change results unless you ask them to.

| variable | effect |
| --- | --- |
| `LLAMA_CASCADE=c,k` | enable the cascade with those counts |
| `LLAMA_CASCADE_VOCAB=path` | enable the vocabulary cut with that id list |
| `LLAMA_CASCADE_SKIP=ffn,gate,norm,add,attn,conv` | drop whole node families (timing only) |
| `CASCADE_CUDA_SKIP=sel0,up,sel1,down,reduce` | disable one kernel (timing only) |
| `CASCADE_CUDA_TIME=1` | per-kernel CUDA event timings, printed per token |
| `CASCADE_CUDA_SLICES=n` | split the down gather across `n` slices (default 32, max 64) |
| `CASCADE_CUDA_EXTRA_DISP=n` | repeat the selection n times to price one dispatch |
| `CASCADE_CUDA_SELPHASE=n` | stop the selection after n radix passes (timing only) |
| `GGML_CUDA_TIME_NODES=1` | per-op timings for the first two graphs |
| `LLAMA_CASCADE_ABLATE=n` | Metal: bitmask that skips selection / up (timing only) |

## Layout

- `ggml/src/ggml-cuda/cascade.cu`, `cascade.cuh` — CUDA kernels and dispatch
- `ggml/src/ggml-metal/kernels/cascade.metal` — Metal kernels
- `src/models/lfm3.cpp`, `lfm-base.cpp` — the LFM3 FFN hook and the cascade builder
- `src/llama-model.cpp` — the unit-major int8 repack, the head cut and the id map

## Status and licence

Research code, kept for reproducibility; the numbers above are the reason it exists. The llama.cpp part
is MIT as upstream. Model weights produced with it are derivatives of
[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) and carry that model's licence, not MIT.
