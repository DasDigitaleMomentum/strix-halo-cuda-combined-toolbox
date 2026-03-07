#!/usr/bin/env bash

set -e

TOOLBOX_NAME="llama-rocm-cuda"
IMAGE_NAME="localhost/strix-halo-cuda-combined:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADDITIONAL_FLAGS="--device /dev/dri --device /dev/kfd --device nvidia.com/gpu=all --group-add video --group-add render --security-opt seccomp=unconfined"

# --- helpers ---

function usage() {
  echo "Usage: $0 <command>"
  echo ""
  echo "Commands:"
  echo "  build         Build the container image (uses cache)"
  echo "  rebuild       Re-clone llama.cpp, reuse cached OS layers"
  echo "  full-rebuild  Full rebuild from scratch (no cache at all)"
  echo "  create        Create the distrobox (image must exist)"
  echo "  rm            Remove the distrobox"
  echo "  all           Build image + create distrobox"
  echo ""
  echo "Image:     $IMAGE_NAME"
  echo "Distrobox: $TOOLBOX_NAME"
  exit 1
}

function check_deps() {
  for cmd in podman distrobox; do
    if ! command -v "$cmd" > /dev/null; then
      echo "Error: '$cmd' is not installed." >&2
      exit 1
    fi
  done
}

function distrobox_exists() {
  distrobox list --no-color 2>/dev/null | tail -n +2 | awk '{print $3}' | grep -qx "$1"
}

# --- commands ---

function cmd_build() {
  echo "Building image: $IMAGE_NAME"
  podman build --build-arg "CACHEBUST=1" -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Build complete: $IMAGE_NAME"
}

function cmd_rebuild() {
  echo "Rebuilding image (fresh llama.cpp clone): $IMAGE_NAME"
  podman build --build-arg "CACHEBUST=$(date +%s)" -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Rebuild complete: $IMAGE_NAME"
}

function cmd_full_rebuild() {
  echo "Full rebuild (no cache at all): $IMAGE_NAME"
  podman build --no-cache -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo "Full rebuild complete: $IMAGE_NAME"
}

function cmd_create() {
  if ! podman image exists "$IMAGE_NAME"; then
    echo "Error: Image $IMAGE_NAME not found. Run '$0 build' first." >&2
    exit 1
  fi

  if distrobox_exists "$TOOLBOX_NAME"; then
    echo "Removing existing distrobox: $TOOLBOX_NAME"
    distrobox rm --force "$TOOLBOX_NAME"
  fi

  echo "Creating distrobox: $TOOLBOX_NAME"
  distrobox create -n "$TOOLBOX_NAME" --image "$IMAGE_NAME" --additional-flags "$ADDITIONAL_FLAGS"
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

# --- main ---

check_deps

case "${1:-}" in
  build)
    cmd_build
    ;;
  rebuild)
    cmd_rebuild
    ;;
  full-rebuild)
    cmd_full_rebuild
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
