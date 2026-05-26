FROM docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /chatter

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Shallow-clone chatterbox (deep clone is too slow under QEMU emulation)
ARG CHATTERBOX_BRANCH=faster
RUN git clone --depth 1 --branch ${CHATTERBOX_BRANCH} \
    https://github.com/rsxdalv/chatterbox.git /tmp/chatterbox

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python package manager and dependencies, clean all caches in same layer
RUN pip install --no-cache-dir uv && \
    uv sync --locked --no-editable --no-cache \
      --no-install-package tts-webui-chatterbox-tts && \
    uv pip install --no-cache --no-deps /tmp/chatterbox && \
    rm -rf /tmp/chatterbox /root/.cache/uv /root/.cache/pip

COPY docker_init.py ./

# Download model weights and clean HF download cache
RUN /chatter/.venv/bin/python docker_init.py && \
    find /root/.cache/huggingface -name "*.incomplete" -delete 2>/dev/null; \
    rm -rf /root/.cache/huggingface/hub/.locks

COPY fatterbox ./fatterbox

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["/chatter/.venv/bin/python", "-m", "fatterbox"]
