---
type: planning
entity: phase
plan: "rocmpin-7.2.4"
phase: 1
status: in_progress
created: "2026-07-21"
updated: "2026-07-21"
---

# Phase 1: Dockerfile auf ROCm 7.2.4 RPM umstellen

> Part of [rocmpin-7.2.4](../plan.md)

## Objective

Das `Dockerfile` von ROCm Nightly-Tarballs auf stabile ROCm 7.2.4 RPM-Pakete umstellen. CUDA 13.1 und Multi-Stage-Build bleiben erhalten.

## Contribution to Plan Goal

Der einzige Build des Projekts (`Dockerfile`) wird von instabilen Nightly-Tarballs auf die getestete ROCm 7.2.4 RPM-Installation umgestellt — der gleiche Ansatz, der im Referenz-Projekt stabil läuft.

## Scope

### Includes

- ROCm 7.2.4 RPM-Repo (`repo.radeon.com/rocm/rhel10/7.2.4/main`) statt S3-Tarballs
- Build-Deps via dnf: `rocm-llvm`, `rocm-device-libs`, `hip-runtime-amd`, `hip-devel`, `rocblas`, `rocblas-devel`, `hipblas`, `hipblas-devel`, `rocm-cmake`
- Runtime-Deps via microdnf: `hip-runtime-amd`, `rocblas`, `hipblas`, `rocminfo`, `radeontop`
- ROCm-Pfade: `/opt/rocm` (Standard-RPM-Pfad) statt `/opt/rocm-7.0`
- Profile-Script `rocm.sh` mit korrekten ENV-Variablen (Bugfix gegenüber Referenz)
- Fedora 43 als Builder-Basis (wie Referenz, da ROCm RPMs darauf getestet)

### Excludes (deferred to later phases)

- Nichts — einzige Phase

## Prerequisites

- [x] Analyse: Referenz-Repo Dockerfile.rocm-7.2.4 eingesehen
- [x] Analyse: Commit ccb0c34 (`-funsafe-math-optimizations`) ist in llama.cpp master

## Deliverables

- [ ] `Dockerfile` — umgeschrieben mit ROCm 7.2.4 RPM
- [ ] Profile-Script mit korrekten ROCm-ENV-Variablen
- [ ] Entfernt: S3-Tarball-Download, `/opt/rocm-7.0`-Pfade, `-DCMAKE_HIP_FLAGS` Workaround

## Acceptance Criteria

- [ ] `docker build --no-cache` läuft durch
- [ ] `llama-cli --help` zeigt beide Backends (ROCm + CUDA)
- [ ] `llama-cli -m <model> -ngl 99 -p "test" -n 5` produziert Output
- [ ] `rocminfo` erkennt GPUs
- [ ] `nvidia-smi` (falls NVIDIA-GPU vorhanden)

## Dependencies on Other Phases

Keine — einzige Phase.

## Notes

- Der `-funsafe-math-optimizations` Flag wird automatisch von llama.cpps CMake gesetzt (seit Commit ccb0c34, 9. Juli 2026)
- Die Referenz hat einen Bug im Profile-Script (leeres `printf`), der hier korrigiert wird
- `gcc14-c++` bleibt als CUDA-Host-Compiler für Kompatibilität
