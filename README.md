# leffm-llamacpp

**leffm** — Liquid Efficient Foundation Model — is LFM3-1.7B with a sparse two-stage FFN cascade and a
cut vocabulary.

A research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), branched at `7fce543`, that runs
it. Everything else is upstream
llama.cpp, unmodified, and nothing here is proposed for upstream. The goal was to measure what a
-30% FLOPs model actually buys in wall clock, not just on paper.

## Results

Batch-1 decode, LFM3-1.7B Q8_0, `llama-bench -p 0 -n 32 -r 3`, best of five. "dense" is stock llama.cpp,
"leffm" is this fork with the cascade and the vocabulary cut.

| backend | dense | leffm | speedup |
| --- | ---: | ---: | ---: |
| CUDA (L4) | 9.01 ms/token | **7.54 ms/token** | **1.19x** |
| Metal (M5) | 15.16 ms/token | **12.47 ms/token** | **1.22x** |
| CPU (M5, `-t 4`) | **22.8 ms/token** | 25.4 ms/token | 0.90x |

The two changes reduce:

- **30.0% of FLOPs per token** — the FFN costs 65.6% of its dense equivalent, the output head 52.4%;
- **7.4% of parameters** through the vocabulary cut.

The GPU speedup is smaller than the FLOP reduction: at batch 1 this decode is bandwidth-bound, and the
exact top-k selections cost real time, so 30% fewer FLOPs is ~16% less wall clock on the L4. On CPU the
fork stays below dense: the naive port measured 0.87x, because ggml's int8 matvec already runs at the
part's limit, so reading fewer bytes buys nothing there. Reusing the score pass's silu instead of a
second expf per selected row gained ~0.03 (0.875 -> 0.901 paired). Replacing both `nth_element`
selections with a radix select measured neutral (`nth_element` ~77 us/layer, radix ~77): an earlier
version of it read 1.00-1.05x because it discarded every bucket above the selection threshold, which
selects the wrong units and still passes the `c = k = F` identity gate — the CPU arm produced
degenerate text at `k = 2304` while that gate passed. With the selection made exact the same node
measures 0.896 (0.852 / 0.896 / 0.905 over three paired runs), and the bench now cross-checks the
selection against `nth_element` at the operating point before timing anything.

### Quality trade-off

Perplexity on a held-out slice of the same corpus (64 sequences, 115,159 bytes), after one matched
continued-pretraining run of 2.67M tokens — dense and leffm trained for the same 163 steps:

| arm | PPL/token | bits/byte |
| --- | ---: | ---: |
| stock LFM3-1.7B | 3.392 | 0.4270 |
| dense, after CPT | **2.832** | **0.3639** |
| leffm | **3.035** | **0.3885** |

Against the dense model at matched training, leffm costs **+0.069 nats (+6.8% bits/byte)**. The two
tokenizations differ slightly (the cut tokenizer needs
1.004x as many tokens for the same text), so bits/byte is the comparable column. No benchmark suite
beyond perplexity was run — treat these as the price on this corpus, not as a general statement.
At `LLAMA_CASCADE=7680,7680` the cascade reproduces dense greedy output exactly, which is the check
that everything below the selection is arithmetically the dense path.

## Usage

Copy-paste, CUDA first, Metal second.

```sh
# NVIDIA GPU: FlashAttention is built in, and `native` compiles for the local GPU
git clone https://github.com/styalai/leffm-llamacpp && cd leffm-llamacpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

# Apple silicon or CPU
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf --local-dir .
```

`-DCMAKE_CUDA_ARCHITECTURES=native` targets the GPU in the machine; set it explicitly only to
cross-compile (`89` Ada — L4, L40S, RTX 40xx; `80` A100; `86` A10 / RTX 30xx; `90` H100). The CUDA
numbers above were measured with `-DGGML_CUDA_FA=OFF`, which changes the attention kernels only.

A converted cascade checkpoint carries its operating point in the GGUF (`lfm3.cascade.c` /
`lfm3.cascade.k`), so the cascade needs no flags, no vocabulary file and no `--no-repack` — the repack
is disabled by the file itself:

```sh
./build/bin/llama-cli -m leffm-1.7b-Q8_0.gguf -p "The quick brown fox" -n 32
```

Such a model is already vocabulary-cut and its tokenizer lives in the GGUF, so `LLAMA_CASCADE_VOCAB`
does not apply. `LLAMA_CASCADE=0,0` turns the cascade off on the same weights (dense FFN), and
`LLAMA_CASCADE="c,k"` overrides the file's operating point.

To reproduce the table above from the **stock 128k** model, where the cut happens at load time and the
cascade is not in the file, the flags are explicit:

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
On that path `--no-repack` and Q8_0 weights are required; a checkpoint that carries `lfm3.cascade.c/k`
disables the repack itself. Omit `LLAMA_CASCADE` (or set `0,0`) for the stock dense model, and
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

- **Tested**: greedy-identity at `c = k = F` on CUDA, Metal and CPU; the radix selection cross-checked
  against `nth_element` (randomized trials, plus real activations through `LLAMA_CASCADE_VERIFY_SEL`,
  which the CPU bench runner runs at `c=5120, k=2304` before it times anything); every kernel variant listed in
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
