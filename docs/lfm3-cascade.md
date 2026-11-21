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

## Performance by backend

Decode at batch 1, LFM3-1.7B Q8_0, `--no-repack`, cascade `c = 5,120`, `k = 2,304` of 7,680. t/s from
`llama-bench -p 0 -n 32 -r 3` (best of five; ms/token derived), CPU from `llama-cli -t 4` median of
three.

| backend | hardware | dense | cascade | cascade + vocab cut | verdict |
| --- | --- | --- | --- | --- | --- |
| CUDA | L4, 23 GB, sm_89, CUDA 13.0 | 110.9 t/s (9.01 ms) | 123.7 (8.09) | **132.7 (7.54)** | **1.19x** |
| Metal | Apple M5, 8 GPU cores, 147.5 GB/s measured | 66.0 t/s (15.16 ms) | 75.1 (13.32) | **80.2 (12.47)** | **1.22x** |
| CPU | x86/ARM, ggml int8 SIMD, `-t 4` | **43.9 t/s (22.8 ms)** | 36.8 (27.2) | 39.3 (25.4) | **0.90x, loses** |

The CPU row stays below dense: the naive port measured 0.87x (dense 43.3 vs cut 39.9), because
ggml's dense int8 matvec already streams bytes at close to what the part can do, so reading fewer of
them buys nothing and the selection is pure overhead. Two changes were tried. Reusing the score pass's
silu instead of a second `expf` per selected row gained ~0.03 (0.901 paired, and the `expf` was nearly
free on this part, so the gain is bookkeeping). Replacing both `nth_element` selections with the radix
select the other backends use is neutral on this CPU: back-to-back on the same machine, `nth_element`
measures 73-83 us/layer and the radix 75-79, with the down op identical once both select the same
units. The cut arm now reads 0.896 (0.852 / 0.896 / 0.905 over three paired interleaved runs, dense
43.6-44.7 t/s).

An earlier revision of the radix read 1.00-1.05x. That version discarded every bucket above the
threshold digit at each pass, so it selected a systematically lower set: the CPU arm produced
degenerate text at `k = 2304` while the `c = k = F` identity gate — which selects everything and
therefore exercises no selection at all — still passed. With the descent keeping the above-threshold
elements the selection is exact (cross-checked against `nth_element` on random data and on real
activations via `LLAMA_CASCADE_VERIFY_SEL`, which the bench runs before it times anything), and the
row is the 0.90 above. The lesson is in the runner: identity at full width does not validate a
selection.

A unit-major down layout measured worse (0.847 paired) and was reverted: the chunk-major layout it
replaced is four near-sequential 128-byte streams per thread, not the scattered lines it was first
misread as, and with a correct selection the down cost depends on which units are gathered (the
deferred per-block scale flush), so a layout A/B must be taken against the same selection. The GPU
rows win for the opposite reason — their gathers reach 160-210 GB/s against a ~300 GB/s array, so
halving the bytes does help, just not proportionally.

### Where the time goes (CUDA, per token, 24 layers)

| kernel | ms/token |
| --- | --- |
| selection 1, gate top-c | 0.9 |
| up gather | 1.3 |
| selection 2, activation top-k | 0.3 |
| down gather | 0.7 |
| partial-sum reduction | 0.2 |
| everything else: gate, attention, shortconv, norms, head | 5.5 |

Dense spends 3.24 ms in its FFN; the cascade spends 2.51 on 52% of the bytes. The gathers run at
~194 GB/s (up) and ~161 GB/s (down) against dense's 236 GB/s sequential matvec, which is why 30% fewer
FLOPs is 16% less wall clock and not 30%.

### Metal specifics

The selection was the whole regression there as well: replacing the original 32-round bisection with
the same radix select moved cascade from 68.2 to 75.1 t/s (+10%) and the cut arm from 72.7 to 80.2.
After that port, ablating both selections changes throughput by less than the noise (74.6 vs 75.1), so
the gathers are the remaining cost and the selection is effectively free. The head gather reaches
142 GB/s of the machine's 147.5 GB/s measured streaming peak.

An earlier session measured parity +-5% on Metal, but that machine was contended (paired ratios swung
0.82-1.24 within minutes); the figures above are best-of-five on a quiet machine and supersede it.

### Quality

One matched continued-pretraining run (2.67M tokens, code/chat corpus, perplexity only, one seed),
evaluated on a held-out slice of the same corpus (64 sequences, 115,159 bytes):

| arm | PPL/token | bits/byte |
| --- | ---: | ---: |
| stock LFM3-1.7B, no training | 3.392 | 0.4270 |
| dense, after CPT | **2.832** | **0.3639** |
| cut + cascade, no training | 3.684 | 0.4563 |
| cut + cascade, after CPT | **3.035** | **0.3885** |

Against dense at matched training, the cut + cascade costs **+0.069 nats (+6.8% bits/byte)**;
zero-shot, before any training, +0.082 nats. The cut tokenizer needs 1.004x as many tokens for the
same text, so bits/byte is the comparable column. That is the price of the recipe on that corpus,
not a general claim.

## What has been tested

- **Correctness, every backend**: greedy continuation identical to dense at `c = k = F` (CUDA and
  Metal), and at 24/24 tokens for `c/k = 2560/1536`, `5120/2304` and `7680/7680` on CPU. The
  full-FFN point is the check that the gather path is arithmetically the dense path.
- **CUDA kernels**: radix select vs the 32-round bisection; scores in registers vs 32 KB of dynamic
  shared memory; private histogram copies; `float4` score fetch; shared-memory bank layouts; 16-byte
  vs 4-byte up loads; one vs two candidates per warp; the down gather with 1, 4 and 8 units in flight;
  4 outputs per thread; one block covering `D` versus 512-byte granules; 16, 32 and 64 down slices;
  and a fully fused selection+gather+reduction, which measured slower and was reverted.
- **Dispatch cost**: 11.5 us per launch, measured by repeating an idempotent selection.
- **Metal**: bisection vs radix select; down threadgroup of 128 vs 512 threads; two units per
  simdgroup with `uint2` loads; the unit-major int8 repack; 16 threadgroups; ablation masks.
- **CPU**: exact selection (nth_element, then the radix select shared with CUDA and Metal), SIMD
  gathered dot, silu reuse from the score pass, both down layouts (chunk-major kept, unit-major
  reverted on a paired regression), thread sweeps from 1 to 8.

## What has not been tested

- CUDA on anything but an L4 (`sm_89`): no A100/H100, no second architecture, single device only.
- FlashAttention was **off** in every CUDA benchmark here. It changes nothing at 32-token contexts,
  but the absolute t/s is not the default-config number.
- Batch 1 only. The cascade applies to `T = 1`; prompt processing and any batched decode take the
  dense path, and no server or concurrent-request behaviour was measured.
- Metal on anything but an 8-core M5, and the down still launches 16 threadgroups (32 and 64 were not
  retried after the scratch was enlarged to allow them).
- Quality beyond perplexity on one corpus: no downstream benchmarks, no second seed, and the
  continued-pretraining budget is 2.67M tokens.
- CUDA graphs: llama.cpp captures by default, and nothing here was tuned for it beyond staying
  capturable.

## Running it

Requires an LFM3 model whose FFN and head match the ids you feed it, and Q8_0 up/down weights.

```sh
# CUDA (sm_89 shown; FlashAttention off keeps the build short and changes nothing at 32 tokens)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_FA=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

# Metal: the default build on macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-cli

# a converted cascade checkpoint: operating point in the GGUF, repack disabled by the file itself
./build/bin/llama-cli -m lfm3-1.7b-cascade-30-Q8_0.gguf -p "..." -n 32

# the stock 128k model: dense baseline, then the cascade, then the cascade with the id cut
./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "..." -n 32 --no-repack
LLAMA_CASCADE=5120,2304 ./build/bin/llama-cli -m ... --no-repack
LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=trim_ids.txt ./build/bin/llama-cli -m ... --no-repack
```

A converted cascade checkpoint writes `lfm3.cascade.c` and `lfm3.cascade.k` (from its
`cascade_config.json`), so the runtime needs no `LLAMA_CASCADE` at all; the variable still overrides
the file, and `LLAMA_CASCADE=0,0` runs the same weights with the dense FFN. Those files also skip the
Q8_0 repack automatically, because the cascade reads the stored layout.

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
