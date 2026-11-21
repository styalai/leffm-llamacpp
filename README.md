# leffm-llamacpp

A research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), branched at `7fce543`, that runs
LFM3-1.7B with a sparse two-stage FFN cascade and a cut vocabulary. Everything else is upstream
llama.cpp, unmodified, and nothing here is proposed for upstream. The goal was to measure what a
-30% FLOPs model actually buys in wall clock, not just on paper.

## Results

Batch-1 decode, LFM3-1.7B Q8_0, `llama-bench -p 0 -n 32 -r 3`, best of five. "dense" is stock llama.cpp,
"leffm" is this fork with the cascade and the vocabulary cut.

| backend | dense | leffm | speedup |
| --- | ---: | ---: | ---: |
| CUDA (L4) | 9.01 ms/token | **7.54 ms/token** | **1.19x** |
| Metal (M5) | 15.16 ms/token | **12.47 ms/token** | **1.22x** |
| CPU (M5, `-t 4`) | 22.6 ms/token | **22.5 ms/token** | **1.00x** |

The two changes reduce:

- **30.0% of FLOPs per token** — the FFN costs 65.6% of its dense equivalent, the output head 52.4%;
- **7.4% of parameters** through the vocabulary cut.

The GPU speedup is smaller than the FLOP reduction: at batch 1 this decode is bandwidth-bound, and the
exact top-k selections cost real time, so 30% fewer FLOPs is ~16% less wall clock on the L4. On CPU the
fork is level with dense: the naive port measured 0.87x, because ggml's int8 matvec already runs at the
part's limit, so reading fewer bytes buys nothing there — reusing the score pass's silu and replacing
both nth_element selections with a radix select brought it to 1.00-1.05x across paired runs.

### Quality trade-off

Perplexity on a held-out slice of the same corpus (64 sequences, 115,159 bytes), after one matched
continued-pretraining run of 2.67M tokens — dense and leffm trained for the same 163 steps:

| arm | PPL/token | bits/byte |
| --- | ---: | ---: |
| stock LFM3-1.7B, no training | 3.392 | 0.4270 |
| dense, after CPT | **2.832** | **0.3639** |
| leffm, no training | 3.684 | 0.4563 |
| leffm, after CPT | **3.035** | **0.3885** |

Against the dense model at matched training, leffm costs **+0.069 nats (+6.8% bits/byte)**; zero-shot,
before any training, **+0.082 nats**. The two tokenizations differ slightly (the cut tokenizer needs
1.004x as many tokens for the same text), so bits/byte is the comparable column. No benchmark suite
beyond perplexity was run — treat these as the price on this corpus, not as a general statement.
At `LLAMA_CASCADE=7680,7680` the cascade reproduces dense greedy output exactly, which is the check
that everything below the selection is arithmetically the dense path.

## Usage

Copy-paste; on macOS drop the `GGML_CUDA` flags and it builds Metal instead.

```sh
git clone https://github.com/styalai/leffm-llamacpp && cd leffm-llamacpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_FA=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf --local-dir .
```

Run leffm (cascade 5120/2304 + vocabulary cut):

```sh
LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=docs/lfm3-trim-ids.txt \
  ./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "The quick brown fox" -n 32 --no-repack
```

Benchmark it:

```sh
LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=docs/lfm3-trim-ids.txt \
  ./build/bin/llama-bench -m LFM3-1.7B-Q8_0.gguf -p 0 -n 32 -r 3 -ngl 999 --no-warmup
```

`docs/lfm3-trim-ids.txt` is the 65,536-id set the numbers above were measured with. The head and tied
embedding are trimmed **at load time** from the stock 128k weights, so no custom model file is needed.
`--no-repack` and Q8_0 weights are required. Omit `LLAMA_CASCADE` for the stock dense model, or
`LLAMA_CASCADE_VOCAB` for the cascade without the cut.

## How it works

A gated FFN computes

$$\mathrm{FFN}(x) = \big(\operatorname{SiLU}(xW_{\text{gate}}^T) \odot (xW_{\text{up}}^T)\big) W_{\text{down}}.$$

The gate must see every unit — it is the router — but the two projections do not:

1. score all `F` units by `|silu(g)|`, keep the top `c` **exactly**;
2. run the up projection for those `c` units only;
3. rescore by the true activation `|silu(g)·u|`, keep the top `k` exactly;
4. run the down projection for those `k` units only.

Per-token cost is $d(F+c+k)$ instead of $3dF$; at `c = 5,120`, `k = 2,304` of `F = 7,680` that is
65.6% of the dense FFN.

Selection is exact, never a threshold: a four-pass radix select over the floating-point bit pattern,
on all three backends. Thresholds cannot hold a per-token count and were measured to prune far less
than they claim.

### Vocabulary cut

The tied input embedding and output head are trimmed at load time to a calibration-derived id set —
65,536 of 128,000 ids, so the head costs 52.4% of the original. `CASCADE_EXPAND` scatters the trimmed
logits back into the full vocabulary with a `-inf` sentinel, so sampling code is untouched. Dropped ids
carry 0.41% of the stock model's probability mass on held-out text and the rebuilt tokenizer costs
1.004x as many tokens for the same bytes.

The cascade applies to single-token steps only (`T = 1`); prompt processing takes the dense path.

## Tested / not tested

- **Tested**: greedy-identity at `c = k = F` on CUDA, Metal and CPU; every kernel variant listed in
  [docs/lfm3-cascade.md](docs/lfm3-cascade.md) (radix vs bisection, load widths, occupancy shapes,
  slice counts, and a fused selection+gather+reduction that measured slower and was reverted);
  per-dispatch cost, measured by repeating an idempotent selection.
- **Not tested**: quality beyond perplexity on one corpus, downstream benchmarks, second seeds.

## Licence

The llama.cpp part is MIT, as upstream. Model weights produced with this fork are derivatives of
[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) and carry that model's licence, not
MIT.

Per-backend internals, per-kernel budgets and every diagnostic knob:
**[docs/lfm3-cascade.md](docs/lfm3-cascade.md)**.
