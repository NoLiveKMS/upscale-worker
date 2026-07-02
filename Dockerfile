FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential wget curl ca-certificates git \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install basicSR dependencies (tb-nightly is needed to prevent basicsr logging errors)
RUN pip install --no-cache-dir tb-nightly

# Install realesrgan, gfpgan, and other requirements
RUN pip install --no-cache-dir \
    realesrgan gfpgan opencv-python-headless \
    runpod>=1.7.0 numpy

# Patch basicsr to support newer torchvision versions (fix functional_tensor import error)
RUN find /usr/local -name "degradations.py" -exec sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/g' {} + 2>/dev/null || true

# Clone official CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer

# Environment variable for caching model weights
ENV WEIGHTS_DIR="/app/weights"
ENV PYTHONUNBUFFERED=1
ENV UPSCALE_MODEL="RealESRGAN_x4plus" \
    UPSCALE_SCALE="4" \
    FACE_RESTORE="gfpgan" \
    TILE_SIZE="400"

# Pre-download all models (including facexlib face detection/parsing models)
COPY download_models.py .
RUN python3 download_models.py && rm download_models.py

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
