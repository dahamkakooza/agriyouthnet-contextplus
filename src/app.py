
import io, torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from model import AgriYouthNetContextPlus, CLASS_NAMES

app = FastAPI(title="AgriYouthNet-Context+ API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AgriYouthNetContextPlus().to(device)
model.load_state_dict(torch.load("results/models/agriyouth_contextplus.pt",
                                 map_location=device))
model.eval()

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@app.get("/")
def root():
    return {"status": "ok", "service": "AgriYouthNet-Context+"}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  crop: float = Form(...),
                  wet: int = Form(...),
                  severe: int = Form(...),
                  on_stem: int = Form(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    c = torch.tensor([[crop, wet, severe, on_stem]],
                     dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x, c), 1)[0].cpu().numpy()
    idx = int(p.argmax())
    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(p[idx]),
        "top3": sorted(
            [{"class": CLASS_NAMES[i], "prob": float(p[i])}
             for i in range(len(CLASS_NAMES))],
            key=lambda d: -d["prob"])[:3],
    }
