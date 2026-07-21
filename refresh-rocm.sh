#!/usr/bin/env bash

set -e

IMAGE_NAME="localhost/llama-rocm:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLBOX_NAME="llama-rocm"
FLAGS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"

# ── Auto-detect container runtime ──
detect_runtime() {
    if command -v docker &>/dev/null; then
        echo "docker"
    elif command -v podman &>/dev/null; then
        echo "podman"
    else
        echo "ERROR: Neither docker nor podman found." >&2
        exit 1
    fi
}
RUNTIME=$(detect_runtime)
echo "Container runtime: $RUNTIME"

# ── helpers ──────────────────────────────────────────────────────

function usage() {
  echo "Usage: $0 <command>"
  echo ""
  echo "Commands:"
  echo "  build         Build the ROCm-only image (uses cache)"
  echo "  rebuild       Re-clone llama.cpp, reuse cached layers, recreate distrobox"
  echo "  full-rebuild  Full rebuild from scratch (no cache), recreate distrobox"
  echo "  create        Create the distrobox (image must exist)"
  echo "  rm            Remove the distrobox"
  echo "  all           Build image (cached) + create distrobox"
  echo ""
  echo "Runtime: $RUNTIME"
  echo "Image:    $IMAGE_NAME"
  echo "Toolbox:  $TOOLBOX_NAME"
  exit 1
}

function check_deps() {
  if ! command -v distrobox &> /dev/null; then
    echo "Error: 'distrobox' is not installed." >&2
    exit 1
  fi
}

function distrobox_exists() {
  distrobox list --no-color 2>/dev/null | tail -n +2 | awk '{print $3}' | grep -qx "$1"
}

# ── commands ────────────────────────────────────────────────────

function cmd_build() {
  echo "Building image with $RUNTIME: $IMAGE_NAME"
  $RUNTIME build -f "$SCRIPT_DIR/Dockerfile.rocm" -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Build complete: $IMAGE_NAME"
}

function cmd_rebuild() {
  echo "Rebuilding image (fresh llama.cpp clone): $IMAGE_NAME"
  $RUNTIME build -f "$SCRIPT_DIR/Dockerfile.rocm" --build-arg "CACHEBUST=$(date +%s)" -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Rebuild complete: $IMAGE_NAME"
}

function cmd_full_rebuild() {
  echo "Full rebuild (no cache at all): $IMAGE_NAME"
  $RUNTIME build -f "$SCRIPT_DIR/Dockerfile.rocm" --no-cache -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Full rebuild complete: $IMAGE_NAME"
}

function cmd_create() {
  if ! $RUNTIME image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Error: Image $IMAGE_NAME not found. Run '$0 build' first." >&2
    exit 1
  fi

  if distrobox_exists "$TOOLBOX_NAME"; then
    echo "Removing existing distrobox: $TOOLBOX_NAME"
    distrobox rm --force "$TOOLBOX_NAME"
  fi

  echo "Creating distrobox: $TOOLBOX_NAME"
  # shellcheck disable=SC2086
  distrobox create -n "$TOOLBOX_NAME" --image "$IMAGE_NAME" --additional-flags "$FLAGS"
  echo "Done. Enter with: distrobox enter $TOOLBOX_NAME"
}

function cmd_rm() {
  if distrobox_exists "$TOOLBOX_NAME"; then
    echo "Removing distrobox: $TOOLBOX_NAME"
    distrobox rm --force "$TOOLBOX_NAME"
    echo "Done"
  else
    echo "$TOOLBOX_NAME not installed, nothing to remove"
  fi
}

# ── main ────────────────────────────────────────────────────────

check_deps

COMMAND="${1:-}"

case "$COMMAND" in
  build)
    cmd_build
    ;;
  rebuild)
    cmd_rebuild
    cmd_create
    ;;
  full-rebuild)
    cmd_full_rebuild
    cmd_create
    ;;
  create)
    cmd_create
    ;;
  rm)
    cmd_rm
    ;;
  all)
    cmd_build
    cmd_create
    ;;
  *)
    usage
    ;;
esac
