# =============================================================================
# Image Upscaling & Face Restoration Worker — RunPod Serverless
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    realesrgan gfpgan opencv-python-headless \
    runpod>=1.7.0 numpy

COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV UPSCALE_MODEL="RealESRGAN_x4plus" \
    UPSCALE_SCALE="4" \
    FACE_RESTORE="gfpgan" \
    TILE_SIZE="400"

CMD ["python3", "-u", "handler.py"]
