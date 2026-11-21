# llama.cpp

> Research fork, branched at `7fce543`, for one experiment. Nothing here is proposed for upstream.

## What this fork adds: LFM3 cascade FFN + vocabulary cut

A gated FFN does `silu(x·W_gateᵀ) ⊙ (x·W_upᵀ) · W_down`. The gate must see every unit — it is the
router — but the projections need not:

1. score all `F` units by `|silu(g)|`, keep the top `c` **exactly**;
2. run the up projection for those `c` units only;
3. rescore by the true activation `|silu(g)·u|`, keep the top `k` exactly;
4. run the down projection for those `k` units only.

Cost per token is `d·(F + c + k)` instead of `3·d·F`: at `c = 5,120`, `k = 2,304` of `F = 7,680` the
FFN costs **65.6% of dense**. The tied embedding and head are additionally cut to a calibration-derived
id set — 65,536 of 128,000 ids puts the head at **52.4%** — with `CASCADE_EXPAND` scattering the trimmed
logits back into the full vocabulary so sampling is untouched. Together: **-30.0% FLOPs/token, -7.4%
parameters** for LFM3-1.7B.

Selection is exact on every backend (a four-pass radix select on the float bit pattern on CUDA and
Metal), not a threshold: thresholds cannot hold a per-token count and were measured to prune far less
than they claim.

### Measured, batch-1 decode, LFM3-1.7B Q8_0, `--no-repack`

| backend | hardware | dense | cascade | cascade + vocab cut |
| --- | --- | --- | --- | --- |
| CUDA | L4 23 GB, sm_89 | 110.9 t/s (9.01 ms) | 123.7 (8.09) | **132.7 (7.54), 1.19x** |
| Metal | Apple M5, 8 cores | 66.0 t/s (15.16 ms) | 75.1 (13.32) | **80.2 (12.47), 1.22x** |
| CPU | x86/ARM int8, `-t 4` | 43.3 t/s (23.1 ms) | 37.8 (26.5) | 39.9 (25.1), **0.87x** |

The CPU row loses: ggml's dense int8 matvec already streams near the part's limit, so reading fewer
bytes buys nothing and the selection is overhead. The GPUs win because their gathers run below the
array roofline, which is also why 30% fewer FLOPs is ~16% less wall clock rather than 30% (per-kernel
budget: selection 0.9 + up 1.3 + selection 0.3 + down 0.7 + reduce 0.2 ms/token on the L4).

Quality, one 2.67M-token continued-pretraining run, perplexity only: after matched training the cut +
cascade is **+6.8% bits/byte** against a dense model trained the same number of steps, +0.082 nats
zero-shot. Correctness: at `LLAMA_CASCADE=7680,7680` the cascade reproduces dense greedy output exactly.

### Tested / not tested

- Tested: greedy-identity at `c = k = F` on CUDA, Metal and CPU; every kernel variant listed in the
  docs below (radix vs bisection, load widths, occupancy shapes, slice counts, and a fused
  selection+gather+reduction that measured slower and was reverted); dispatch cost.
- Not tested: any GPU but an L4 (no A100/H100, one architecture), FlashAttention **off** in every CUDA
  benchmark, batch 1 only (prompts take the dense path), Metal on anything but an 8-core M5, and
  quality beyond perplexity on one corpus.

Build and run instructions, the full per-backend detail and every diagnostic knob:
**[docs/lfm3-cascade.md](docs/lfm3-cascade.md)**.

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
