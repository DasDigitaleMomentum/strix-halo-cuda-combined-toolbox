# Strix Halo Dual-GPU ROCm + CUDA llama.cpp Toolbox

A distrobox-based container running **llama.cpp** compiled with both **AMD ROCm/HIP** and **NVIDIA CUDA** backends, enabling dual-GPU inference on systems with an AMD APU and a discrete NVIDIA GPU.

Built for the **AMD Ryzen AI MAX+ 395** (Radeon 8060S, gfx1151) + **NVIDIA GeForce RTX 3060** (Ampere, compute 8.6), but adaptable to other hardware by changing the build targets.

## Prerequisites

- **Linux host** (tested on Ubuntu 25.10)
- **AMD GPU** with ROCm support (Radeon 8060S / gfx1151 or similar)
- **NVIDIA GPU** with CUDA support (RTX 3060 or similar)
- **podman** (>= 5.x)
- **distrobox** (>= 1.8)
- **nvidia-container-toolkit** with CDI configured:
  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```
- NVIDIA driver installed on host (tested with 590.48.01)

## Build

```bash
podman build -t strix-halo-cuda-combined .
```

Build time is significant (~20-40 minutes) due to compiling llama.cpp with both HIP and CUDA backends.

## Create Distrobox

```bash
./refresh-toolbox.sh
```

This will build the image (if not already built) and create a distrobox named `llama-rocm-cuda` with the required device passthrough flags:

```
--device /dev/dri
--device /dev/kfd
--device nvidia.com/gpu=all
--group-add video
--group-add render
--security-opt seccomp=unconfined
```

Options:
- `./refresh-toolbox.sh --build` — force rebuild the image before creating the distrobox
- `./refresh-toolbox.sh rm` — remove the distrobox

## Usage

```bash
distrobox enter llama-rocm-cuda
```

### Verify devices

```bash
llama-cli --list-devices
```

Expected output:
```
CUDA0: NVIDIA GeForce RTX 3060 (12.00 GiB)
ROCm0: AMD Radeon 8060S Graphics (126.72 GiB)
```

### Run inference

Single GPU (AMD):
```bash
llama-cli -m model.gguf --device ROCm0 -ngl 99 -p "Hello" -n 128
```

Single GPU (NVIDIA):
```bash
llama-cli -m model.gguf --device CUDA0 -ngl 99 -p "Hello" -n 128
```

Dual-GPU with tensor split (e.g., 90% AMD / 10% NVIDIA):
```bash
llama-cli -m model.gguf --device ROCm0 -ngl 99 --tensor-split 9,1 -p "Hello" -n 128
```

### VRAM estimator

```bash
gguf-vram-estimator.py model.gguf
```

## Key Build Decisions

| Decision | Rationale |
|---|---|
| **Fedora 42 builder** / Fedora 43 runtime | Avoids glibc 2.41 + CUDA `cudafe++` `rsqrt noexcept` clash in builder stage |
| **`GGML_BACKEND_DL=ON`** | Required for multi-backend (HIP+CUDA) — backends are loaded as separate `.so` plugins at runtime |
| **`GGML_NATIVE=OFF` + `GGML_CPU_ALL_VARIANTS=ON`** | CPU backend compiled for multiple ISAs; avoids build-host-specific optimizations |
| **`CMAKE_CUDA_ARCHITECTURES=86`** | Targets RTX 3060 (Ampere, compute capability 8.6) |
| **`AMDGPU_TARGETS=gfx1151`** | Targets AMD Radeon 8060S (Strix Halo iGPU) |
| **`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14`** | GCC 15 (Fedora 42 default) is incompatible with nvcc; gcc14-c++ is installed explicitly |
| **`HIP_VISIBLE_DEVICES=0`** | Required ENV for HIP runtime to detect AMD GPU when CUDA toolkit is also present |
| **`--device nvidia.com/gpu=all`** | CDI passthrough for NVIDIA GPU devices into the container |

## Benchmark Results

Tested with **Qwen3-30B-A3B Q4_K_M** (MoE, sparse model):

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| ROCm0 only | 1183 | 73.2 |
| 90% ROCm0 / 10% CUDA0 | 1165 | 73.4 |
| 70% ROCm0 / 30% CUDA0 | 1184 | 73.4 |

For this sparse MoE model, tensor-split shows no performance gain since the AMD iGPU has ample VRAM. Dual-GPU is primarily useful for **dense models** that exceed single-GPU VRAM or for running **parallel inference** across GPUs.

## Known Limitations

- **Cross-backend `-ot` (override-tensor) is NOT supported**: Placing individual tensors on different backends (e.g., `-ot ".*ffn_gate_exps.*=CUDA0"`) causes an abort. Only `--tensor-split` (layer-level split) works across backends.
- **Build targets are hardware-specific**: The default build targets `gfx1151` (AMD) and compute `86` (NVIDIA Ampere). For other GPUs, modify `AMDGPU_TARGETS` and `CMAKE_CUDA_ARCHITECTURES` in the Dockerfile.
- **Large build**: The multi-stage build pulls ROCm and CUDA toolchains, resulting in a large builder layer. The runtime image is significantly smaller.

## Adapting for Other Hardware

Edit the Dockerfile and change:
- `AMDGPU_TARGETS=gfx1151` — set to your AMD GPU's GFX target (e.g., `gfx1100` for RX 7900 XTX)
- `CMAKE_CUDA_ARCHITECTURES=86` — set to your NVIDIA GPU's compute capability (e.g., `89` for RTX 4090, `90` for H100)

## License

llama.cpp is licensed under MIT. This container build configuration is provided as-is.
