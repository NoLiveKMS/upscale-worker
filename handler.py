"""
RunPod Serverless Image Upscaling & Face Restoration Handler.

Combines Real-ESRGAN (upscaling) with GFPGAN or CodeFormer (face restoration).
"""

import base64
import os
import sys

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
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")

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
CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
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
    half=torch.cuda.is_available(),
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
elif FACE_RESTORE == "codeformer":
    # Add CodeFormer to the path so we can import its modules
    import sys
    sys.path.insert(0, "/app/CodeFormer")
    from basicsr.utils.registry import ARCH_REGISTRY
    import basicsr.archs.codeformer_arch  # force registration
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"]
    ).to(device)
    
    checkpoint = torch.load(get_model_path(CODEFORMER_URL), map_location=device)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()
    
    face_restorer = net
    print("CodeFormer face restorer loaded.")

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
        "face_enhance": true,
        "codeformer_fidelity": 0.5
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
    codeformer_fidelity = float(job_input.get("codeformer_fidelity", 0.75))

    # Decode image
    try:
        # Strip Data URL prefix if present (e.g. data:image/png;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[-1]
            
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
            if FACE_RESTORE == "gfpgan":
                _, _, output = face_restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5,
                )
            elif FACE_RESTORE == "codeformer":
                from facelib.utils.face_restoration_helper import FaceRestoreHelper
                from basicsr.utils import img2tensor, tensor2img
                from torchvision.transforms.functional import normalize
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                face_helper = FaceRestoreHelper(
                    upscale_factor=scale,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model="retinaface_resnet50",
                    save_ext="png",
                    use_parse=True,
                    device=device,
                )
                
                face_helper.clean_all_results()
                face_helper.read_image(img)
                face_helper.get_face_landmarks_and_shift(save_half_face=False, only_center_face=False)
                face_helper.align_warp_faces()
                
                for cropped_face in face_helper.cropped_faces:
                    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output_face_t = face_restorer(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                        restored_face = tensor2img(output_face_t, rgb2bgr=True, min_max=(-1, 1))
                        
                    face_helper.add_restored_face(restored_face)
                    
                face_helper.get_inverse_affine(None)
                output = face_helper.paste_faces_to_input_image(
                    save_ext="png",
                    upscale_factor=scale,
                    bg_upsampler=upsampler
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
