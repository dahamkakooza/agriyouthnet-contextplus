"""
AgriYouthNet-Context+ FastAPI service.

Serves the trained cross-attention model as a REST endpoint.

Run from the src/ folder:

    python -m uvicorn app:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the Swagger UI.
"""

import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# The model class lives next to this file. We import it directly, which works
# when uvicorn is launched from src/ (uvicorn adds cwd to sys.path).
# ---------------------------------------------------------------------------
from model import AgriYouthNetContextPlus, CLASS_NAMES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
CHECKPOINT = BASE_DIR / "results" / "models" / "agriyouth_contextplus.pt"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Maps a numeric crop id to something readable in the response.
CROP_LABELS = {0.0: "maize", 0.5: "tomato", 1.0: "potato"}


# ---------------------------------------------------------------------------
# Model loading — done once at import time so every request is fast
# ---------------------------------------------------------------------------
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at {CHECKPOINT}.\n"
        f"Download it from Colab with:\n"
        f"  from google.colab import files\n"
        f"  files.download('/content/drive/MyDrive/agriyouth-demo/results/models/"
        f"agriyouth_contextplus.pt')\n"
        f"then place it in {CHECKPOINT.parent}."
    )

model = AgriYouthNetContextPlus(fusion="attention", n_heads=4).to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgriYouthNet-Context+ API",
    version="0.1.0",
    description=(
        "Cross-attention multimodal plant disease classification. "
        "Send a leaf image and four context fields, get the top-3 diagnosis."
    ),
)

# The React Native client and the Gradio demo both run on different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    """Simple check that the service is alive."""
    return {
        "status": "ok",
        "service": "AgriYouthNet-Context+",
        "device": DEVICE,
        "classes": len(CLASS_NAMES),
    }


@app.get("/health", tags=["meta"])
def health():
    """Deeper health check — reports checkpoint and model info."""
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "status": "healthy",
        "device": DEVICE,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_mb": round(CHECKPOINT.stat().st_size / 1e6, 3),
        "params": n_params,
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Leaf photo (JPEG or PNG)"),
    crop: float = Form(..., description="0.0=maize, 0.5=tomato, 1.0=potato"),
    wet: int = Form(..., description="1 if wet season, 0 otherwise"),
    severe: int = Form(..., description="1 if symptoms look severe, 0 otherwise"),
    on_stem: int = Form(..., description="1 if symptoms on the stem, 0 otherwise"),
):
    """
    Diagnose a plant disease from a leaf image + farmer-reported context.

    Returns the top class, its confidence, and the top-3 ranked list.
    """
    # ---- Validate the upload ------------------------------------------------
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Use JPEG or PNG.",
        )

    # ---- Read and preprocess the image -------------------------------------
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x = preprocess(img).unsqueeze(0).to(DEVICE)

    # ---- Build the context vector ------------------------------------------
    ctx = torch.tensor(
        [[float(crop), float(wet), float(severe), float(on_stem)]],
        dtype=torch.float32,
    ).to(DEVICE)

    # ---- Run inference ------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(x, ctx)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())

    # ---- Assemble the response ---------------------------------------------
    top3 = sorted(
        [
            {
                "class": CLASS_NAMES[i],
                "class_display": CLASS_NAMES[i].replace("_", " "),
                "prob": float(probs[i]),
            }
            for i in range(len(CLASS_NAMES))
        ],
        key=lambda d: -d["prob"],
    )[:3]

    return {
        "prediction": {
            "class": CLASS_NAMES[top_idx],
            "class_display": CLASS_NAMES[top_idx].replace("_", " "),
            "confidence": float(probs[top_idx]),
        },
        "top3": top3,
        "context_received": {
            "crop": CROP_LABELS.get(crop, str(crop)),
            "wet_season": bool(wet),
            "severe_symptoms": bool(severe),
            "on_stem": bool(on_stem),
        },
    }