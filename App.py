"""
Super-Resolution API — FastAPI backend
Loads the trained Generator and serves /predict endpoint.

Usage:
    pip install fastapi uvicorn pillow torch torchvision python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torchvision.transforms.functional as TF

# ── Model Architecture (must match training) ─────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.ps   = nn.PixelShuffle(2)
        self.act  = nn.PReLU()

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, in_channels=3, feat=64, num_res=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, feat, 9, 1, 4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(feat) for _ in range(num_res)])
        self.post_res = nn.Sequential(
            nn.Conv2d(feat, feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat)
        )
        self.upsample = nn.Sequential(UpsampleBlock(feat), UpsampleBlock(feat))
        self.output = nn.Sequential(
            nn.Conv2d(feat, in_channels, 9, 1, 4),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.initial(x)
        res  = self.res_blocks(feat)
        res  = self.post_res(res) + feat
        up   = self.upsample(res)
        return self.output(up)


# ── TTA Helper ────────────────────────────────────────────────────────────────

def tta_predict(model, lr_tensor, device):
    preds = []
    for flip in [False, True]:
        for k in [0, 1, 2, 3]:
            aug = lr_tensor.clone()
            if flip:
                aug = torch.flip(aug, dims=[3])
            if k > 0:
                aug = torch.rot90(aug, k=k, dims=[2, 3])
            with torch.no_grad():
                out = model(aug)
            if k > 0:
                out = torch.rot90(out, k=(4 - k), dims=[2, 3])
            if flip:
                out = torch.flip(out, dims=[3])
            preds.append(out)
    return torch.stack(preds).mean(dim=0)


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Plant Leaf Super-Resolution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"best_generator.pth"
model = None


@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}. /predict will return 503.")
        return
    model = Generator().to(DEVICE)
    # NEW — weights_only=False needed for older .pth files, suppresses warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"Model loaded from {MODEL_PATH} on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), use_tta: bool = True):
    if model is None:
        raise HTTPException(503, "Model not loaded. Check MODEL_PATH env variable.")

    # Read and validate image
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not parse image file.")

    # Resize to exactly 32×32 if needed
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BICUBIC)

    # Preprocess: [0,1] → [-1,1]
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)  # [1,3,32,32]
    lr_tensor = lr_tensor * 2.0 - 1.0

    # Inference
    if use_tta:
        fake_hr = tta_predict(model, lr_tensor, DEVICE)
    else:
        with torch.no_grad():
            fake_hr = model(lr_tensor)

    # Postprocess: [-1,1] → [0,255]
    fake_hr_255 = ((fake_hr.clamp(-1, 1) + 1) / 2 * 255)
    fake_hr_255 = fake_hr_255.round().clamp(0, 255).to(torch.uint8)
    img_np = fake_hr_255.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert to PNG bytes
    out_img = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png",
                              headers={"Content-Disposition": "attachment; filename=upscaled.png"})