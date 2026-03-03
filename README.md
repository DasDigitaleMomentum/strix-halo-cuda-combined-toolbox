# Strix Halo Dual-GPU ROCm + CUDA llama.cpp Toolbox

A standalone, distrobox-based container that runs [llama.cpp](https://github.com/ggerganov/llama.cpp) with both AMD ROCm/HIP and NVIDIA CUDA backends compiled into a single build. Which is supported by llama.cpp since a few month.
Both GPUs are visible simultaneously, enabling dual-GPU inference via `--tensor-split` on systems with an AMD Strix Halo APU and a discrete NVIDIA GPU.

### Background

I forked [kyuz0's amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) to have a clean, isolated place to experiment with combined llama.cpp backends. The reason: I bought an [NVMe-to-OCuLink adapter](https://www.amazon.de/GINTOOYUN-PCI-E4-0-Adapterkabel-weibliches-Hostkabel/dp/B0F8N34H1R) and a [Minisforum DEG1 graphics dock](https://minisforumpc.eu/products/minisforum-deg1-grafik-docking-station) to hook up my old RTX 3060 alongside the Strix Halo's integrated Radeon 8060S. My idea was throw both GPUs at inference and see what happens.

Getting there wasnt simple. ROCm and CUDA have different toolchain requirements, and making them coexist in a single llama.cpp build turned into small rabbit hole of glibc version clashes, GCC incompatibilities with nvcc, HIP/CUDA symbol collisions, and runtime detection issues where one stack would shadow the other. 
This repo is the result of that process.

## How it works

llama.cpp is built with `GGML_BACKEND_DL=ON`, which compiles each backend (HIP, CUDA, CPU variants) as a separate `.so` plugin loaded at runtime. This avoids symbol clashes between the ROCm and CUDA toolchains that would occur with static linking. The result is a single `llama-cli` binary that can address both `ROCm0` and `CUDA0` devices.

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
| `./refresh-toolbox.sh build` | Build (or rebuild) the container image |
| `./refresh-toolbox.sh create` | Create the distrobox from the local image |
| `./refresh-toolbox.sh rm` | Remove the distrobox |
| `./refresh-toolbox.sh all` | Build image + create distrobox |

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

Qwen3-30B-A3B Q4_K_M (MoE, sparse model):

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| ROCm0 only | 1183 | 73.2 |
| 90% ROCm0 / 10% CUDA0 | 1165 | 73.4 |
| 70% ROCm0 / 30% CUDA0 | 1184 | 73.4 |

For this sparse MoE model, tensor-split shows no speedup because the AMD iGPU has ample VRAM and the model's sparse access pattern doesn't benefit from offloading. Dual-GPU tensor-split is primarily useful for **dense models** that exceed single-GPU memory or for running **parallel inference** workloads.

## Known limitations

- **Cross-backend `-ot` (override-tensor) is not supported.** Placing individual tensors on a different backend (e.g., `-ot ".*ffn_gate_exps.*=CUDA0"`) causes an abort -- the backend cannot run operations on tensors allocated by another backend. Only `--tensor-split` (layer-level split) works across backends.
- **MoE models see no speedup from tensor-split.** Sparse expert access means only a fraction of parameters are active per token, so splitting across GPUs adds communication overhead without reducing computation.
- **Build targets are hardware-specific.** The defaults target gfx1151 (AMD) and compute 86 (NVIDIA). Other GPUs require changing the Dockerfile build args.

## Adapting for other hardware

Edit the Dockerfile and change these two cmake variables:

- `AMDGPU_TARGETS=gfx1151` -- your AMD GPU's GFX target (e.g., `gfx1100` for RX 7900 XTX, `gfx1103` for 780M)
- `CMAKE_CUDA_ARCHITECTURES=86` -- your NVIDIA GPU's compute capability (e.g., `89` for RTX 4090, `90` for H100, `120` for RTX 5090)

## License

MIT -- see [LICENSE](LICENSE).
