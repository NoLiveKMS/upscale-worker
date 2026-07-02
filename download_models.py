import os
import urllib.request
import torch
from gfpgan import GFPGANer

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

models = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
}

print(f"Starting model weights pre-downloads to {WEIGHTS_DIR}...")
for name, url in models.items():
    path = os.path.join(WEIGHTS_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"{name} already exists.")

# Trigger automatic download of facexlib weights (face detection & parsing models)
# by initializing GFPGANer on CPU during image build.
print("Pre-downloading facexlib weights (face detection & parsing)...")
try:
    GFPGANer(
        model_path=os.path.join(WEIGHTS_DIR, "GFPGANv1.3.pth"),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cpu"
    )
    print("Facexlib weights downloaded successfully.")
except Exception as e:
    print(f"Error during GFPGANer CPU initialization: {e}")

print("All models successfully cached.")
