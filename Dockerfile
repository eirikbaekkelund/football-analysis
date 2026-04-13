FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
COPY annotation_tool/ ./annotation_tool/

# Install torchkick + annotation + model deps
# PyTorch CUDA index is already declared in pyproject.toml [tool.uv]
RUN uv pip install --system -e ".[annotation,training,reid,onnx]"
# cache installations
RUN uv pip cache purge

EXPOSE 8080

# Single worker — required to share in-memory state and GPU model singletons
CMD ["uvicorn", "annotation_tool.server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
