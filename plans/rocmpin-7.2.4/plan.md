---
type: planning
entity: plan
plan: "rocmpin-7.2.4"
status: active
created: "2026-07-21"
updated: "2026-07-21"
---

# Plan: ROCm Nightly → 7.2.4 RPM (CUDA Dual-Build)

## Problem / Context

Das kombinierte `Dockerfile` (CUDA + ROCm Dual-Backend) nutzt **ROCm Nightly-Tarballs** von `therock-nightly-tarball.s3.amazonaws.com`. Diese sind instabil und verursachen einen Crash beim Start von llama.cpp. Das Referenz-Projekt [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) zeigt, dass **ROCm 7.2.4 via RPM** stabil läuft.

## Target Outcome

`Dockerfile` liefert wieder einen lauffähigen Dual-Backend-Build (ROCm 7.2.4 + CUDA 13.1), der sowohl auf Strix Halo (gfx1151) als auch Radeon R9700 (gfx1201) funktioniert.

## Guiding Decisions & Constraints

- ROCm-Quelle: AMD RPM-Repo (`repo.radeon.com/rocm/rhel10/7.2.4/main`)
- CUDA bleibt: 13.1 via NVIDIA RPM-Repo
- Build-Basis: Fedora 43 (getestet im Referenz-Projekt mit ROCm 7.2.4)
- GPU-Targets: `gfx1151;gfx1201` (iGPU + dGPU)
- Multi-Stage-Build bleibt (Builder → Runtime)
- Ghostty-Integration und Profile-Scripts bleiben erhalten
- `-funsafe-math-optimizations` wird automatisch von llama.cpps CMake gesetzt (Commit ccb0c34)

### Scope-Bounding Assumptions

- Der CUDA-Repo für Fedora 42 funktioniert auch auf Fedora 43 (laut bestehendem Kommentar im Dockerfile)
- g++-14 als CUDA-Host-Compiler verhindert weiterhin den cudafe++ noexcept-Clash

## Requirements

### Functional

- [ ] ROCm 7.2.4 wird vom AMD RPM-Repo installiert
- [ ] CUDA 13.1 wird vom NVIDIA RPM-Repo installiert
- [ ] llama.cpp buildet mit `-DGGML_HIP=ON -DGGML_CUDA=ON`
- [ ] Ergebnis-Image startet llama-server/cli ohne Crash

### Non-Functional

- [ ] Build-Zeit nicht wesentlich länger als vorher
- [ ] Image bleibt multi-stage (Runtime minimal)

## Scope

### In Scope

- `Dockerfile` umschreiben: ROCm-Quelle von Nightly-Tarballs auf 7.2.4 RPM
- Profile-Script `rocm.sh` fixen (Referenz-Bug: leeres Script)
- Entfernen: S3-Tarball-Download, `/opt/rocm-7.0` Pfade, `-DCMAKE_HIP_FLAGS` Workaround
- Build testen

### Out of Scope

- `Dockerfile.rocm` (separater ROCm-only Build) ändern
- `Dockerfile.pytorch` ändern
- `refresh-toolbox.sh` Logik ändern

## Definition of Done

- [ ] `Dockerfile` baut erfolgreich mit `docker build --no-cache`
- [ ] `llama-cli --help` läuft ohne Crash
- [ ] `llama-cli -m <model> -ngl 99 -p "test" -n 5` produziert Output
- [ ] ROCm-Devices werden korrekt erkannt (rocminfo)

## Testing Strategy

- [ ] `docker build --no-cache -t localhost/strix-halo-cuda-combined:test .` 
- [ ] Container starten und `llama-cli` mit Modell testen
- [ ] `rocminfo` auf GPU-Erkennung prüfen

## Phases

| Phase | Title | Contribution | Detail | Status |
|-------|-------|--------------|--------|--------|
| 1 | Dockerfile auf ROCm 7.2.4 RPM umstellen | Stabiles ROCm, CUDA bleibt | [Phase](phases/phase-1.md) | in_progress |

## Risks & Open Questions

| Risk/Question | Impact | Mitigation/Answer |
|---------------|--------|-------------------|
| ROCm 7.2.4 RPMs (rhel10) auf Fedora 43 inkompatibel | Build schlägt fehl | Getestet im Referenz-Projekt — läuft |
| cudafe++ noexcept-Clash mit Fedora 43 glibc | CUDA-Compilation bricht ab | g++-14 als Host-Compiler (bewährte Lösung) |
| gfx1201 Support in 7.2.4 | dGPU nicht nutzbar | gfx1201 ist in AMDGPU_TARGETS seit ROCm 6.x |

## Changelog

### 2026-07-21

- Plan created
