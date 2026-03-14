# Image Upscaling & Face Restoration RunPod Worker

RunPod Serverless worker for AI image upscaling and face restoration:

- **Real-ESRGAN** — 2x/4x upscaling for photos, anime, illustrations
- **GFPGAN** — Face restoration and enhancement

Combined pipeline: upscale background + restore faces in one pass.

## Input / Output

### Input
```json
{
  "input": {
    "image": "<base64 encoded image>",
    "scale": 4,
    "face_enhance": true
  }
}
```

### Output
```json
{
  "image": "<base64 upscaled PNG>",
  "format": "png",
  "scale": 4,
  "original_size": [512, 512],
  "output_size": [2048, 2048]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSCALE_MODEL` | `RealESRGAN_x4plus` | Model: `RealESRGAN_x4plus`, `RealESRGAN_x2plus`, `RealESRGAN_x4plus_anime_6B` |
| `FACE_RESTORE` | `gfpgan` | Face restoration: `gfpgan` or `none` |
| `TILE_SIZE` | `400` | Tile size (lower = less VRAM, 0 = no tiling) |

## License

- Real-ESRGAN: BSD-3-Clause
- GFPGAN: Apache 2.0
