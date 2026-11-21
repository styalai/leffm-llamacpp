# llama.cpp

> Research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), branched at `7fce543`. One
> experiment: a sparse LFM3 FFN plus a vocabulary cut. Nothing here is proposed for upstream.

## Performance

LFM3-1.7B Q8_0, batch-1 decode, `llama-bench -p 0 -n 32 -r 3`, best of five. "basic" is stock dense
llama.cpp, "leffm" is this fork with the cascade and the vocabulary cut.

| backend | hardware | basic | leffm | speedup |
| --- | --- | --- | --- | --- |
| CUDA | L4 23 GB, sm_89 | 9.01 ms/token | **7.54 ms/token** | **1.19x** |
| Metal | Apple M5, 8 GPU cores | 15.16 ms/token | **12.47 ms/token** | **1.22x** |
| CPU | x86/ARM int8, `-t 4` | 23.1 ms/token | 25.1 ms/token | 0.87x |

**-30.0% FLOPs/token** (FFN at 65.6% of dense, head at 52.4%) and **-7.4% parameters**. On GPUs that is
1.19-1.22x wall clock; on CPU it is a slowdown, because there ggml's dense int8 matvec already runs at
the part's limit while a GPU gather does not — which is also why 30% fewer FLOPs is ~16% less time on an
L4 rather than 30%.

Quality: **+6.8% bits/byte** after a matched 2.67M-token continued-pretraining run (one corpus,
perplexity only), +0.082 nats zero-shot. At `LLAMA_CASCADE=7680,7680` the cascade reproduces dense greedy
output exactly.

## Usage

Copy-paste; on macOS drop the `GGML_CUDA` flags and it builds Metal instead.

```sh
git clone https://github.com/styalai/leffm-llamacpp && cd leffm-llamacpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_FA=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench

hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf --local-dir .
```

Dense, then the cascade, then the cascade with the vocabulary cut:

```sh
./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "The quick brown fox" -n 32 --no-repack

LLAMA_CASCADE=5120,2304 \
  ./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "The quick brown fox" -n 32 --no-repack

LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=docs/lfm3-trim-ids.txt \
  ./build/bin/llama-cli -m LFM3-1.7B-Q8_0.gguf -p "The quick brown fox" -n 32 --no-repack
```

Same three arms as a benchmark:

```sh
for env in "" "LLAMA_CASCADE=5120,2304" \
           "LLAMA_CASCADE=5120,2304 LLAMA_CASCADE_VOCAB=docs/lfm3-trim-ids.txt"; do
  env $env ./build/bin/llama-bench -m LFM3-1.7B-Q8_0.gguf -p 0 -n 32 -r 3 -ngl 999 --no-warmup
done
```

`docs/lfm3-trim-ids.txt` is the 65,536-id set the numbers above were measured with. The head and tied
embedding are trimmed **at load time** from the stock 128k weights, so no custom model file is needed.
`--no-repack` is required, and Q8_0 weights are required for the cascade.

## How it works

A gated FFN does `silu(x·W_gateᵀ) ⊙ (x·W_upᵀ) · W_down`. The gate must see every unit — it is the
router — but the projections need not:

1. score all `F` units by `|silu(g)|`, keep the top `c` **exactly**;
2. run the up projection for those `c` units only;
3. rescore by the true activation `|silu(g)·u|`, keep the top `k` exactly;
4. run the down projection for those `k` units only.

Cost per token is `d·(F + c + k)` instead of `3·d·F`. Selection is exact on every backend (four-pass
radix select on the float bit pattern on CUDA and Metal), not a threshold: thresholds cannot hold a
per-token count and were measured to prune far less than they claim. The vocabulary cut trims the tied
head, and `CASCADE_EXPAND` scatters the trimmed logits back into the full vocabulary so sampling code is
untouched. It applies to single-token steps only; prompts take the dense path.

Build details, per-backend internals and every diagnostic knob:
**[docs/lfm3-cascade.md](docs/lfm3-cascade.md)**.

### What was and was not tested

- Tested: greedy-identity at `c = k = F` on CUDA, Metal and CPU; every kernel variant in the docs
  (radix vs bisection, load widths, occupancy shapes, slice counts, and a fused
  selection+gather+reduction that measured slower and was reverted); per-dispatch cost.
- Not tested: any GPU but an L4 (no A100/H100), FlashAttention **off** in every CUDA benchmark, batch 1
  only, Metal on anything but an 8-core M5, quality beyond perplexity on one corpus.

The llama.cpp part is MIT as upstream; model weights produced with it are derivatives of
[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) and carry that model's licence.

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
