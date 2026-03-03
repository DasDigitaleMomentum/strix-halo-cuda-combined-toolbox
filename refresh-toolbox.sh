#!/usr/bin/env bash

set -e

# Dual-GPU ROCm + CUDA llama.cpp distrobox
TOOLBOX_NAME="llama-rocm-cuda"
IMAGE_NAME="strix-halo-cuda-combined"
ADDITIONAL_FLAGS="--device /dev/dri --device /dev/kfd --device nvidia.com/gpu=all --group-add video --group-add render --security-opt seccomp=unconfined"

# Check dependencies
for cmd in podman distrobox; do
  if ! command -v "$cmd" > /dev/null; then
    echo "Error: '$cmd' is not installed." >&2
    exit 1
  fi
done

# Helper: check if distrobox exists
function distrobox_exists() {
  distrobox list --no-color 2>/dev/null | tail -n +2 | awk '{print $3}' | grep -qx "$1"
}

# Handle 'rm' command
if [ "${1:-}" = "rm" ]; then
  if distrobox_exists "$TOOLBOX_NAME"; then
    echo "Removing distrobox: $TOOLBOX_NAME"
    distrobox rm --force "$TOOLBOX_NAME"
    echo "Done"
  else
    echo "$TOOLBOX_NAME not installed, nothing to remove"
  fi
  exit 0
fi

# Remove existing distrobox if present
if distrobox_exists "$TOOLBOX_NAME"; then
  echo "Removing existing distrobox: $TOOLBOX_NAME"
  distrobox rm --force "$TOOLBOX_NAME"
fi

# Build image if it doesn't exist or --build is passed
if [ "${1:-}" = "--build" ] || ! podman image exists "localhost/$IMAGE_NAME"; then
  echo "Building image: $IMAGE_NAME"
  podman build -t "$IMAGE_NAME" "$(dirname "$0")"
fi

# Create distrobox
echo "Creating distrobox: $TOOLBOX_NAME (image: localhost/$IMAGE_NAME)"
distrobox create -n "$TOOLBOX_NAME" --image "localhost/$IMAGE_NAME" --additional-flags "$ADDITIONAL_FLAGS"

echo "Done. Enter with: distrobox enter $TOOLBOX_NAME"
