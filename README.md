# LeafLens — Plant Super-Resolution Deployment

## What's Included

```
sr_backend/
  app.py                  ← FastAPI server (serves /predict)
  notebook_save_cell.py   ← Paste this cell into your Kaggle notebook after training

sr_frontend/
  index.html              ← Complete frontend web app (zero dependencies)
```

---

## Step 1 — Add the Save Cell to Your Notebook

Copy the contents of `notebook_save_cell.py` into a new cell **immediately after** the training loop cell (Cell 11 in your notebook). Run it to produce:

- `/kaggle/working/best_generator.pth`   ← best val-MAE weights (already saved by training loop)
- `/kaggle/working/final_generator.pth`  ← final epoch weights
- `/kaggle/working/model_metadata.json`  ← architecture config
- `/kaggle/working/generator_scripted.pt` ← TorchScript export (optional, for CPU inference)

Download `best_generator.pth` from Kaggle output.

---

## Step 2 — Run the Backend Server

```bash
pip install fastapi uvicorn pillow torch torchvision python-multipart

# Place best_generator.pth in the same folder as app.py, then:
MODEL_PATH=best_generator.pth uvicorn app:app --host 0.0.0.0 --port 8000
```

Test the server is alive:
```bash
curl http://localhost:8000/health
```

---

## Step 3 — Open the Frontend

Just open `index.html` in any browser — no build step, no npm.

1. Set the **API URL** field to wherever your server is running (default: `http://localhost:8000`)
2. Drop in any PNG/JPG leaf image (32×32 or it'll be auto-resized)
3. Toggle **Test-Time Augmentation** on for best quality (averages 8 predictions)
4. Hit **Upscale Image**
5. Download the 128×128 PNG result

---

## API Reference

### `POST /predict`

| Field      | Type   | Description |
|------------|--------|-------------|
| `file`     | binary | Multipart image upload (PNG or JPG) |
| `use_tta`  | bool   | Query param — enable 8× TTA (default: true) |

Returns: `image/png` — 128×128 upscaled image

### `GET /health`

Returns JSON with `model_loaded` status and device info.

---

## Architecture Notes

The Generator is an SRGAN-style network:
- **Input**: 32×32 RGB in [-1, 1]
- **Residual blocks**: 16 × ResBlock (Conv-BN-PReLU-Conv-BN + skip)
- **Upsampling**: 2 × PixelShuffle (sub-pixel convolution, 2× each = 4× total)
- **Output**: 128×128 RGB in [-1, 1] → clipped to [0, 255]

TTA averages predictions across 8 geometric transforms (4 rotations × 2 flips), each reversed before averaging — identical to inference used during Kaggle submission.