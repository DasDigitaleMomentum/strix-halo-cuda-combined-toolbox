# build stage — Fedora 42 to avoid glibc 2.41 + cudafe++ noexcept clash
FROM registry.fedoraproject.org/fedora:42 AS builder

# rocm 7.2 repo
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-7.2]
name=ROCm7.2
baseurl=https://repo.radeon.com/rocm/rhel10/7.2/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

# nvidia cuda repo (using fedora42 repo — works on 43)
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

# deps: ROCm + CUDA build tools
RUN dnf -y --nodocs --setopt=install_weak_deps=False \
  --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
  install \
  make gcc gcc-c++ gcc14-c++ cmake lld clang clang-devel compiler-rt libcurl-devel ninja-build \
  rocm-llvm rocm-device-libs hip-runtime-amd hip-devel \
  rocblas rocblas-devel hipblas hipblas-devel rocm-cmake libomp-devel libomp \
  rocminfo radeontop \
  cuda-nvcc-13-1 cuda-cudart-devel-13-1 cuda-driver-devel-13-1 libcublas-devel-13-1 \
  git-core vim sudo rsync \
  && dnf clean all && rm -rf /var/cache/dnf/*

# rocm + cuda env
ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  HIP_CLANG_PATH=/opt/rocm/llvm/bin \
  HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
  CUDA_PATH=/usr/local/cuda \
  PATH=/usr/local/cuda/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64

# llama.cpp
WORKDIR /opt/llama.cpp
ARG REPO=https://github.com/ggerganov/llama.cpp.git
ARG BRANCH=master
RUN git clone -b ${BRANCH} --single-branch --recursive ${REPO} .

# build — dual backend: ROCm/HIP + CUDA
RUN git clean -xdf \
  && git submodule update --recursive \
  && cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -mllvm --amdgpu-unroll-threshold-local=600" \
  -DAMDGPU_TARGETS=gfx1151 \
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
RUN find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/lib64/ \; \
  && ldconfig

# helper
COPY gguf-vram-estimator.py /usr/local/bin/gguf-vram-estimator.py
RUN chmod +x /usr/local/bin/gguf-vram-estimator.py

# runtime stage
FROM registry.fedoraproject.org/fedora-minimal:43

# rocm 7.2 repo
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-7.2]
name=ROCm7.2
baseurl=https://repo.radeon.com/rocm/rhel10/7.2/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

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

# runtime deps: ROCm + CUDA runtime
RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
  --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
  install \
  bash ca-certificates libatomic libstdc++ libgcc libgomp sudo \
  hip-runtime-amd rocblas hipblas \
  rocminfo radeontop procps-ng \
  cuda-cudart-13-1 libcublas-13-1 \
  && microdnf clean all && rm -rf /var/cache/dnf/*

# copy
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/llama.cpp/build/bin/rpc-* /usr/local/bin/

# ld — include CUDA lib path
RUN echo "/usr/local/lib"  > /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/local.conf \
  && ldconfig \
  && cp -n /usr/local/lib/libllama*.so* /usr/lib64/ 2>/dev/null || true \
  && ldconfig

# env for runtime
ENV CUDA_PATH=/usr/local/cuda \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda/bin:/opt/rocm/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64

# helper
COPY gguf-vram-estimator.py /usr/local/bin/gguf-vram-estimator.py
RUN chmod +x /usr/local/bin/gguf-vram-estimator.py

# profile
RUN printf '%s\n' \
  > /etc/profile.d/rocm.sh && chmod +x /etc/profile.d/rocm.sh \
  && echo 'source /etc/profile.d/rocm.sh' >> /etc/bashrc

# shell
CMD ["/bin/bash"]
