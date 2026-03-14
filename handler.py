"""
RunPod Serverless Image Upscaling & Face Restoration Handler.

Combines Real-ESRGAN (upscaling) with GFPGAN or CodeFormer (face restoration).
"""

import base64
import os

import cv2
import numpy as np
import runpod
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("UPSCALE_MODEL", "RealESRGAN_x4plus")
DEFAULT_SCALE = int(os.environ.get("UPSCALE_SCALE", "4"))
FACE_RESTORE = os.environ.get("FACE_RESTORE", "gfpgan")  # gfpgan, codeformer, or none
TILE_SIZE = int(os.environ.get("TILE_SIZE", "400"))

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "num_block": 23, "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x2plus": {
        "num_block": 23, "scale": 2,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "num_block": 6, "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
}

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
WEIGHTS_DIR = "/tmp/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_model_path(url: str) -> str:
    filename = os.path.join(WEIGHTS_DIR, os.path.basename(url))
    if not os.path.isfile(filename):
        print(f"Downloading {url}...")
        torch.hub.download_url_to_file(url, filename)
    return filename


# ---------------------------------------------------------------------------
# Initialize models at startup
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Upscale Worker — Model: {DEFAULT_MODEL}, Face: {FACE_RESTORE}")
print("=" * 60)

config = MODEL_CONFIGS[DEFAULT_MODEL]
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=config["num_block"], num_grow_ch=32, scale=config["scale"],
)

upsampler = RealESRGANer(
    scale=config["scale"],
    model_path=get_model_path(config["url"]),
    model=model,
    tile=TILE_SIZE,
    tile_pad=10,
    pre_pad=0,
    half=True,
)
print(f"Real-ESRGAN ({DEFAULT_MODEL}) loaded.")

face_restorer = None
if FACE_RESTORE == "gfpgan":
    face_restorer = GFPGANer(
        model_path=get_model_path(GFPGAN_URL),
        upscale=config["scale"],
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    print("GFPGAN face restorer loaded.")

print("Ready.")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """
    Input:
    {
        "image": "<base64 encoded image>",
        "scale": 4,
        "face_enhance": true
    }

    Output:
    {
        "image": "<base64 encoded upscaled image>",
        "format": "png",
        "scale": 4,
        "original_size": [width, height],
        "output_size": [width, height]
    }
    """
    job_input = job["input"]

    image_b64 = job_input.get("image", "")
    if not image_b64:
        return {"error": "Missing 'image' field (base64 encoded image)"}

    scale = int(job_input.get("scale", DEFAULT_SCALE))
    face_enhance = job_input.get("face_enhance", FACE_RESTORE != "none")

    # Decode image
    try:
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        return {"error": f"Invalid image data: {e}"}

    if img is None:
        return {"error": "Could not decode image"}

    original_h, original_w = img.shape[:2]

    try:
        if face_enhance and face_restorer is not None:
            _, _, output = face_restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5,
            )
        else:
            output, _ = upsampler.enhance(img, outscale=scale)
    except Exception as e:
        return {"error": f"Upscaling failed: {e}"}

    # Encode output
    _, buffer = cv2.imencode(".png", output)
    output_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "image": output_b64,
        "format": "png",
        "scale": scale,
        "original_size": [original_w, original_h],
        "output_size": [output.shape[1], output.shape[0]],
    }


runpod.serverless.start({"handler": handler})
