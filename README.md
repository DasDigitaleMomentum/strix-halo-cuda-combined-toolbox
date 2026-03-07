# Strix Halo Dual-GPU ROCm + CUDA llama.cpp Toolbox

A standalone, distrobox-based container that runs [llama.cpp](https://github.com/ggerganov/llama.cpp) with both AMD ROCm/HIP and NVIDIA CUDA backends compiled into a single build. Which is supported by llama.cpp since a few month.
Both GPUs are visible simultaneously, enabling dual-GPU inference via `--tensor-split` on systems with an AMD Strix Halo APU and a discrete NVIDIA GPU.

### Background

I forked [kyuz0's amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) to have a clean, isolated place to experiment with combined llama.cpp backends. The reason: I bought an [NVMe-to-OCuLink adapter](https://www.amazon.de/GINTOOYUN-PCI-E4-0-Adapterkabel-weibliches-Hostkabel/dp/B0F8N34H1R) and a [Minisforum DEG1 graphics dock](https://minisforumpc.eu/products/minisforum-deg1-grafik-docking-station) to hook up my old RTX 3060 alongside the Strix Halo's integrated Radeon 8060S. My idea was throw both GPUs at inference and see what happens.

Getting there wasnt simple. ROCm and CUDA have different toolchain requirements, and making them coexist in a single llama.cpp build turned into small rabbit hole of glibc version clashes, GCC incompatibilities with nvcc, HIP/CUDA symbol collisions, and runtime detection issues where one stack would shadow the other. 
This repo is the result of that process.

## How it works

llama.cpp is built with `GGML_BACKEND_DL=ON`, which compiles each backend (HIP, CUDA, CPU variants) as a separate `.so` plugin loaded at runtime. This avoids symbol clashes between the ROCm and CUDA toolchains that would occur with static linking. The result is a single `llama-cli` or `llama-server` or `llama-bench` binary that can address both `ROCm0` and `CUDA0` devices.

Find the adjusted and partially generated README below:

## Prerequisites

- **Linux host** (tested on Ubuntu 25.10)
- **AMD GPU** with ROCm support (tested: Radeon 8060S / gfx1151 on Ryzen AI MAX+ 395)
- **NVIDIA GPU** with proprietary driver installed (tested: RTX 3060 with driver 590.48.01)
- **podman** >= 5.x
- **distrobox** >= 1.8
- **nvidia-container-toolkit** with CDI configured:
  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```

## Quick start

```bash
# Build the image and create the distrobox (takes 20-40 min first time)
./refresh-toolbox.sh all

# Enter the toolbox
distrobox enter llama-rocm-cuda

# Verify both GPUs are visible
llama-cli --list-devices
# CUDA0: NVIDIA GeForce RTX 3060 (12.00 GiB)
# ROCm0: AMD Radeon 8060S Graphics (126.72 GiB)
```

## refresh-toolbox.sh commands

| Command | Description |
|---|---|
| `./refresh-toolbox.sh build` | Build the container image (uses cache — fast, but won't pull new llama.cpp) |
| `./refresh-toolbox.sh rebuild` | Re-clone llama.cpp and rebuild, reuse cached OS/dependency layers |
| `./refresh-toolbox.sh full-rebuild` | Full rebuild from scratch — no cache at all (slowest, use when changing OS deps) |
| `./refresh-toolbox.sh create` | Create the distrobox from the local image |
| `./refresh-toolbox.sh rm` | Remove the distrobox |
| `./refresh-toolbox.sh all` | Build image + create distrobox |

> **Updating llama.cpp:** `build` reuses Podman's build cache, so it won't pick up new llama.cpp commits. Use `rebuild` to pull the latest llama.cpp while keeping cached OS layers, or `full-rebuild` to start completely from scratch.

The distrobox is created with these flags for GPU passthrough:

```
--device /dev/dri --device /dev/kfd --device nvidia.com/gpu=all
--group-add video --group-add render --security-opt seccomp=unconfined
```

You can also build manually if you prefer:

```bash
podman build -t localhost/strix-halo-cuda-combined:latest .
```

## Usage examples

Run on AMD GPU only:
```bash
llama-cli -m model.gguf --device ROCm0 -ngl 99 -p "Hello" -n 128
```

Run on NVIDIA GPU only:
```bash
llama-cli -m model.gguf --device CUDA0 -ngl 99 -p "Hello" -n 128
```

Split layers across both GPUs (90% AMD / 10% NVIDIA):
```bash
llama-cli -m model.gguf --device ROCm0 -ngl 99 --tensor-split 9,1 -p "Hello" -n 128
```

Estimate VRAM requirements for a model:
```bash
gguf-vram-estimator.py model.gguf
```

## Key technical decisions

| Decision | Why |
|---|---|
| Fedora 42 builder, Fedora 43 runtime | Fedora 43 ships glibc 2.41 which triggers a `rsqrt noexcept` clash in `cudafe++`. Fedora 42 (glibc 2.40) avoids this while the runtime stage uses 43 for latest packages. |
| `GGML_BACKEND_DL=ON` | Loads HIP and CUDA as separate `.so` plugins at runtime, avoiding static symbol clashes between the two toolchains. |
| `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` | GCC 15 (Fedora 42 default) is not supported by nvcc. The `gcc14-c++` package is installed explicitly. |
| `HIP_VISIBLE_DEVICES=0` | Without this ENV, the HIP runtime fails to detect the AMD GPU when CUDA libraries are also present. |
| `GGML_NATIVE=OFF` + `GGML_CPU_ALL_VARIANTS=ON` | CPU backend is compiled for multiple ISAs instead of being locked to the build host's CPU features. |
| `AMDGPU_TARGETS=gfx1151` | Strix Halo / Radeon 8060S. Change for your AMD GPU. |
| `CMAKE_CUDA_ARCHITECTURES=86` | Ampere / RTX 3060. Change for your NVIDIA GPU. |

## Benchmark results

All benchmarks below were run with `llama-bench -mmp 0 -fa 1` (mmap disabled, flash attention enabled) using build `2afcdb97` (8185). Hardware: Radeon 8060S (ROCm0, 126.72 GiB UMA) + RTX 3060 (CUDA0, 12 GiB GDDR6).

> **Important: device ordering.** llama.cpp enumerates CUDA0 as Device 0 and ROCm0 as Device 1. In `-ts X/Y`, X is the CUDA0 (RTX 3060) share and Y is the ROCm0 (Radeon) share. This is the opposite of what you might expect if you think of the Radeon as the "primary" GPU.

### Qwen3.5-27B dense (Q4_0, 14.63 GiB)

Direct `--tensor-split` across CUDA0 + ROCm0. Cross-backend layer split works natively.

| Config | pp512 (t/s) | pp2048 (t/s) | pp4096 (t/s) | tg128 (t/s) | PP:512 Δ | PP:2048 Δ | PP:4096 Δ | TG Δ |
|---|---|---|---|---|---|---|---|---|
| **Baseline (ROCm0 only)** | 349.20 ± 3.65 | 347.67 ± 1.45 | 334.04 ± 2.13 | 11.88 ± 0.01 | — | — | — | — |
| **`-ts 3/1` (75% CUDA / 25% ROCm)** | **459.85 ± 5.93** | **580.91 ± 1.56** | **603.84 ± 0.10** | **14.73 ± 0.03** | **+31.7%** | **+67.1%** | **+80.8%** | **+24.0%** |

The RTX 3060 is the primary compute device (75% share) — its GDDR6 bandwidth (360 GB/s) far exceeds the Radeon's shared DDR5 (~100 GB/s effective for GPU). PP improvement scales dramatically with context length.

```bash
# Recommended for Q4_0 (general use)
llama-server -m model-Q4_0.gguf -mmp 0 -fa 1 -ts 3/1
```

### Qwen3.5-27B dense (Q8_0, 26.62 GiB)

Larger quant — the RTX 3060 can only hold ~40% of the model, so the split ratio is inverted.

| Config | pp512 (t/s) | pp2048 (t/s) | pp4096 (t/s) | tg128 (t/s) | PP:512 Δ | PP:2048 Δ | PP:4096 Δ | TG Δ |
|---|---|---|---|---|---|---|---|---|
| **Baseline (ROCm0 only)** | 320.35 ± 1.40 | 317.75 ± 1.25 | 308.83 ± 1.03 | 7.30 ± 0.01 | — | — | — | — |
| **`-ts 1/1.5` (~40% CUDA / 60% ROCm)** | **379.02 ± 1.41** | **484.08 ± 1.25** | **497.91 ± 0.90** | **8.19 ± 0.00** | **+18.3%** | **+52.4%** | **+61.2%** | **+12.2%** |

**`-ts 1/1.5` is the sweet spot** — fills ~10.6 GiB of the RTX 3060's 12 GiB. Going 50/50 exceeds VRAM (~13.3 GiB needed).

```bash
# Recommended for Q8_0
llama-server -m model-Q8_0.gguf -mmp 0 -fa 1 -ts 1/1.5
```

### Row-split vs layer-split

Both `-sm row` (weight matrix split) and `-sm layer` (default, sequential layer split) work cross-backend. Performance is nearly identical:

| Model | Ratio | Mode | PP:512 | PP:2048 | TG:128 |
|---|---|---|---|---|---|
| Q4_0 | 3:1 | layer | 459.85 | 580.91 | **14.73** |
| Q4_0 | 3:1 | row | 459.52 | 581.72 | 14.41 |
| Q8_0 | 1:1.5 | layer | 379.02 | 484.08 | **8.19** |
| Q8_0 | 1:1.5 | row | 376.81 | **484.99** | 8.12 |

Layer-split has a slight TG advantage (0.9-2.4%) due to lower per-token synchronization overhead. Row-split is marginally better at long PP. In practice, either works — **layer-split is the safer default**.

### Qwen3.5-122B-A10B MXFP4 MoE (63.57 GiB, 48 layers, 256 experts/layer)

Direct `--tensor-split` works for MoE models too. The 122B MoE model is ~63.6 GiB, so the RTX 3060 can only hold a small fraction — but even that fraction helps significantly at longer contexts.

| Config | pp512 (t/s) | pp2048 (t/s) | pp4096 (t/s) | tg128 (t/s) | PP:512 Δ | PP:2048 Δ | PP:4096 Δ | TG Δ |
|---|---|---|---|---|---|---|---|---|
| **Baseline (ROCm0 only)** | 344.54 ± 2.79 | 340.48 ± 1.98 | 325.67 ± 3.55 | 19.31 ± 0.01 | — | — | — | — |
| `-ts 1/6` (~14% CUDA) | 357.08 ± 1.11 | 385.98 ± 3.49 | 378.31 ± 3.95 | 19.57 ± 0.01 | +3.6% | +13.4% | +16.2% | +1.3% |
| **`-ts 1/5.2` (~16% CUDA)** | **360.57 ± 1.12** | **393.49 ± 2.34** | **386.52 ± 2.75** | **19.62 ± 0.01** | **+4.7%** | **+15.6%** | **+18.7%** | **+1.6%** |
| `-ts 1/4.5` (~18% CUDA) | OOM | OOM | OOM | OOM | — | — | — | — |

**`-ts 1/5.2` is the sweet spot** — places ~10.2 GiB on the RTX 3060 (just under the 12 GiB limit). Going higher causes OOM because MoE layers are large (~1.32 GiB each with 256 experts). PP improvement scales with context length: +4.7% at 512 tokens, +18.7% at 4096 tokens.

```bash
# Recommended for MoE 122B
llama-server -m model-MXFP4.gguf -mmp 0 -fa 1 -ts 1/5.2
```

## Known limitations

- **Cross-backend `-ot` (override-tensor) is unreliable.** While placing a single layer on the other backend can work (e.g., `-ot "blk.0=CUDA0"`), scaling beyond 1-2 layers fails. The reason: when a backend can't handle certain tensor types or operations (like `f32` norm weights on ROCm), those tensors get silently redirected to the other backend as a fallback. With more layers, the accumulated overflow exhausts the smaller GPU's VRAM (e.g., 529 redirected tensors → OOM on the 12 GiB RTX 3060). Only `--tensor-split` (layer-level split) is reliable across backends.
- **Device ordering is not configurable.** CUDA0 is always Device 0, ROCm0 is Device 1. The `-ts X/Y` ratio assigns X to CUDA0 and Y to ROCm0 — which is counterintuitive when the Radeon has 10x more VRAM. Models that exceed the RTX 3060's 12 GiB share will OOM if the split ratio isn't adjusted.
- **PP:65536+ context is not possible for the 122B MoE model.** The model (63.6 GiB) plus KV cache for 64K context (~8 GiB) exceeds available UMA after OS/desktop overhead. PP:32768 works but is near the limit.
- **Build targets are hardware-specific.** The defaults target gfx1151 (AMD) and compute 86 (NVIDIA). Other GPUs require changing the Dockerfile build args.

## Adapting for other hardware

Edit the Dockerfile and change these two cmake variables:

- `AMDGPU_TARGETS=gfx1151` -- your AMD GPU's GFX target (e.g., `gfx1100` for RX 7900 XTX, `gfx1103` for 780M)
- `CMAKE_CUDA_ARCHITECTURES=86` -- your NVIDIA GPU's compute capability (e.g., `89` for RTX 4090, `90` for H100, `120` for RTX 5090)

## License

MIT -- see [LICENSE](LICENSE).
