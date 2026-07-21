# build stage — Fedora 42 to avoid glibc 2.41 + cudafe++ noexcept clash (e.g. rsqrt)
FROM registry.fedoraproject.org/fedora:42 AS builder

# nvidia cuda repo (fedora42)
RUN <<'EOF'
tee /etc/yum.repos.d/cuda-fedora42.repo <<REPO
[cuda-fedora42-x86_64]
name=cuda-fedora42-x86_64
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64
enabled=1
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/D42D0685.pub
REPO
EOF

# rocm 7.2.4 repo (stable, RPM-based — replaces nightly tarballs)
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-7.2.4]
name=ROCm7.2.4
baseurl=https://repo.radeon.com/rocm/rhel9/7.2.4/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

# deps: build tools + CUDA + ROCm
RUN dnf -y --nodocs --setopt=install_weak_deps=False \
  --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
  install \
  make gcc gcc-c++ gcc14-c++ cmake lld clang clang-devel compiler-rt libcurl-devel ninja-build \
  rdma-core-devel \
  rocm-llvm rocm-device-libs hip-runtime-amd hip-devel \
  rocblas rocblas-devel hipblas hipblas-devel rocm-cmake libomp-devel libomp \
  cuda-nvcc-13-1 cuda-cudart-devel-13-1 cuda-driver-devel-13-1 libcublas-devel-13-1 \
  git-core vim sudo rsync patch rocminfo radeontop \
  && dnf clean all && rm -rf /var/cache/dnf/*

# rocm + cuda env
ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  HIP_CLANG_PATH=/opt/rocm/llvm/bin \
  HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
  CUDA_PATH=/usr/local/cuda \
  PATH=/usr/local/cuda/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib

# llama.cpp
WORKDIR /opt/llama.cpp
ARG REPO=https://github.com/ggerganov/llama.cpp.git
ARG BRANCH=master
ARG CACHEBUST=1
RUN echo "cache-bust: ${CACHEBUST}" && git clone -b ${BRANCH} --single-branch --recursive ${REPO} .

COPY llama-grammar.patch /tmp/llama-grammar.patch

# build — dual backend: ROCm/HIP + CUDA
RUN git clean -xdf \
  && git submodule update --recursive \
  && patch -p1 < /tmp/llama-grammar.patch \
  && cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DAMDGPU_TARGETS="gfx1151;gfx1201" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RPC=ON \
  -DLLAMA_HIP_UMA=ON \
  -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON \
  -DROCM_PATH=/opt/rocm \
  -DHIP_PATH=/opt/rocm \
  -DHIP_PLATFORM=amd \
  && cmake --build build --config Release -- -j$(nproc) \
  && cmake --install build --config Release

# libs
RUN mkdir -p /usr/local/lib64 \
  && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
  && ldconfig

# helper
COPY gguf-vram-estimator.py /usr/local/bin/gguf-vram-estimator.py
RUN chmod +x /usr/local/bin/gguf-vram-estimator.py

# runtime stage
FROM registry.fedoraproject.org/fedora-minimal:43

# nvidia cuda repo (runtime libs only)
RUN <<'EOF'
tee /etc/yum.repos.d/cuda-fedora42.repo <<REPO
[cuda-fedora42-x86_64]
name=cuda-fedora42-x86_64
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64
enabled=1
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/D42D0685.pub
REPO
EOF

# rocm 7.2.4 repo (runtime libs)
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-7.2.4]
name=ROCm7.2.4
baseurl=https://repo.radeon.com/rocm/rhel10/7.2.4/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

# runtime deps: CUDA + ROCm runtime + system tools
RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
  --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
  install \
  bash ca-certificates libatomic libstdc++ libgcc libgomp libibverbs sudo \
  radeontop procps-ng vim \
  cuda-cudart-13-1 libcublas-13-1 \
  hip-runtime-amd rocblas hipblas rocminfo \
  && microdnf clean all && rm -rf /var/cache/dnf/*

# copy llama.cpp build artifacts
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/llama.cpp/build/bin/ggml-rpc-* /usr/local/bin/

# ld — include CUDA + ROCm lib paths
RUN echo "/usr/local/lib"       > /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/lib64"     >> /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/local.conf \
  && echo "/opt/rocm/lib"        >> /etc/ld.so.conf.d/local.conf \
  && echo "/opt/rocm/lib64"      >> /etc/ld.so.conf.d/local.conf \
  && ldconfig \
  && cp -n /usr/local/lib/libllama*.so* /usr/lib64/ 2>/dev/null || true \
  && cp -n /usr/local/lib64/libllama*.so* /usr/lib64/ 2>/dev/null || true \
  && ldconfig

# env for runtime
ENV CUDA_PATH=/usr/local/cuda \
  ROCM_PATH=/opt/rocm \
  HIP_PLATFORM=amd \
  HIP_PATH=/opt/rocm \
  HIP_CLANG_PATH=/opt/rocm/llvm/bin \
  HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
  HIP_VISIBLE_DEVICES=0,1 \
  PATH=/usr/local/cuda/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib

# ghostty shell integration (sourced from host .bashrc via GHOSTTY_RESOURCES_DIR)
RUN mkdir -p /usr/share/ghostty/shell-integration/bash
COPY shell-integration/bash/ /usr/share/ghostty/shell-integration/bash/

# profile — ROCm env for interactive shells
RUN printf '%s\n' \
  'export ROCM_PATH=/opt/rocm' \
  'export HIP_PLATFORM=amd' \
  'export HIP_PATH=/opt/rocm' \
  'export HIP_CLANG_PATH=/opt/rocm/llvm/bin' \
  'export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode' \
  'export PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"' \
  'export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib"' \
  > /etc/profile.d/rocm.sh && chmod +x /etc/profile.d/rocm.sh \
  && echo 'source /etc/profile.d/rocm.sh' >> /etc/bashrc

# redirect ghostty integration to container-local copy when running in ghostty
RUN printf '%s\n' \
  '# If running inside Ghostty, point GHOSTTY_RESOURCES_DIR to the container-local copy' \
  '# so that the host .bashrc "source $GHOSTTY_RESOURCES_DIR/..." line finds the files.' \
  'if [ -n "$GHOSTTY_RESOURCES_DIR" ]; then' \
  '  export GHOSTTY_RESOURCES_DIR=/usr/share/ghostty' \
  'fi' \
  > /etc/profile.d/ghostty.sh && chmod +x /etc/profile.d/ghostty.sh

# shell
CMD ["/bin/bash"]
