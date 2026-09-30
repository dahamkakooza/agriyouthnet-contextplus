# AgriYouthNet-Context+

A small, multimodal plant disease classifier that fuses a leaf photo with a
few pieces of farmer-reported context — using cross-attention — and runs on
the kind of entry-level Android phone that youth agripreneurs in Eastern
Uganda actually own.

This is my BSc Software Engineering capstone at African Leadership University.
Supervisor: Mr Emmanuel Adjei. Submitted September 2026.

---

## Why this exists

Two facts that don't sit well together.

The first: sub-Saharan Africa loses roughly a third of its crop yield to
diseases and pests every year, and smallholder farmers grow about 80% of
the region's food. Uganda has one extension officer for every 5,000
farmers — the recommended ratio is 1 to 500. Most farmers are diagnosing
by eye, and by the time they get it wrong, the harvest is gone.

The second: about 78% of Uganda's population is under 30, rural youth
unemployment is above 13%, and less than half of young people are
engaged in agriculture. The common story is that "youth don't want to
farm." I don't think that's true. I think they don't want the version of
farming they've been handed — no tools, no information, no dignity.

There are already very good CNN models that diagnose plant disease with
99%+ accuracy. The problem is that they are 90 MB to 550 MB. The phones
that youth actually carry have 1–2 GB of RAM and 8–16 GB of storage.
A ResNet-50 will not fit. MobileNetV3 at 5.83 MB technically fits but
leaves no room for anything else on the device.

So the goal of this project is not to make a more accurate model. It is
to make a *small enough* model that is also *accurate enough on the hard
cases* to be worth trusting.

---

## What the model does

Given a leaf image (224×224×3) and a 4-dimensional context vector
(crop, season, symptom severity, symptom location), it predicts one of
ten PlantVillage classes across maize, tomato, and potato.

The contribution is in how the two signals are combined. The baseline
model (AgriYouthNet-Context) just concatenates a 128-dim visual feature
vector with a 32-dim context feature vector and feeds the result to a
classifier. That assumes the two modalities contribute in a fixed way
regardless of the input. In practice they don't — for obvious diseases
the image should dominate, for ambiguous ones the farmer's answer about
the season and symptom location matters more.

AgriYouthNet-Context+ replaces concatenation with Multi-Head Cross-Attention.
Visual features are used as the query, context features as the keys and
values. Each head attends to a different slice of the context, and the
fusion weights adapt per input.

The same class also supports `concat` and `gating` fusion, so the ablation
study can swap mechanisms cleanly without touching anything else.

---

## Architecture
Leaf image (224×224×3) Context vector (4-dim)
│ │
▼ ▼
Visual Backbone Context Encoder
Conv2D + BN + ReLU (32) Dense 32 + ReLU
DepthwiseSep (64) Dense 32 + ReLU
DepthwiseSep (128) │
GAP → 128-d visual │
│ │
└──────────► MHCA (4 heads) ◄──────────┘
Visual Q · Context K,V
│
Add + LayerNorm
│
Dense 10 + Softmax
│
Diagnosis

text

Roughly 0.17 M parameters. ~0.65 MB as Float32. Targets <0.2 MB after
INT8 quantization.

---

## Repo layout
agriyouthnet-contextplus/
├── notebooks/
│ └── AgriYouthNet_ContextPlus.ipynb # end-to-end notebook: EDA, training, ablation, demo
├── src/
│ ├── model.py # architecture (importable, no notebook deps)
│ ├── app.py # FastAPI service with Swagger UI
│ └── gradio_app.py # Gradio MVP
├── results/
│ ├── figures/ # training curves, confusion matrix, ablation plots
│ ├── logs/ # ablation.json with raw numbers
│ └── models/ # trained checkpoint
├── docs/
│ ├── designs/ # Figma wireframe exports
│ └── screenshots/ # app screenshots for the report
├── requirements.txt
├── .gitignore
└── README.md

text

The `data/` folder is not committed. It's 2.7 GB. The notebook downloads
it and filters it in the first few cells.

---

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/agriyouthnet-contextplus.git
cd agriyouthnet-contextplus
pip install -r requirements.txt
If you want GPU, PyTorch 2.2+ is required. The notebook was developed on
a Colab T4. Everything runs on CPU too, just slower.

Running the notebook
The notebook is the main artifact. Open notebooks/AgriYouthNet_ContextPlus.ipynb
in Colab.

You need a Kaggle API token to download PlantVillage. Get one from
your Kaggle account settings,
then in Colab add it as a secret named KAGGLE_API_TOKEN (the 🔑 icon
on the left sidebar).

Then: Runtime → Run all. First cell mounts Drive, third cell pulls
the Kaggle token, and the rest runs top-to-bottom in about 25 minutes on
a T4.

Running the API
bash
uvicorn src.app:app --reload
Swagger UI at http://127.0.0.1:8000/docs. The /predict endpoint takes
a file upload and four form fields, returns the top-3 classes with
confidences.

Running the Gradio MVP
bash
python src/gradio_app.py
Opens a local web UI. Drop a leaf photo, answer four questions, get a
diagnosis. Use share=True in the launch call to get a public URL that
lasts ~72 hours.

Results so far
Trained on a 10-class PlantVillage subset (~18,500 images after
filtering), 80/20 stratified split, ImageNet normalization, Adam with
weight decay 1e-4, StepLR schedule, 10 epochs.

Model	Params	Size	Val Acc
ResNet-50	23.5 M	89.75 MB	97.75%
MobileNetV3	1.53 M	5.83 MB	96.72%
EfficientNet-B0	4.02 M	15.34 MB	98.48%
AgriYouthNet	0.15 M	0.57 MB	95.99%
AgriYouthNet-Context	0.15 M	0.58 MB	99.06%
AgriYouthNet-Context+	~0.17 M	~0.65 MB	see ablation
The baseline numbers come from the earlier AgriYouthNet paper (Kakooza,
2026). Full ablation results are in results/figures/ablation.png and
results/logs/ablation.json.

The ablation checks two things:

Fusion strategy — does attention beat concat and gating at equal
size and equal training budget?

Head count — 1 vs 2 vs 4 vs 8 heads. More heads is not always
better at this parameter scale.

Per-class breakdown (early vs. late blight for both tomato and potato) is
in the notebook. That pair is the interesting one — visually similar,
different treatment, and a wrong diagnosis is expensive.

Designs
The app wireframe (4 screens) is in docs/designs/wireframe.png. The
flow is:

Capture — big camera button, two chips for crop and season

Context form — crop radio, wet/severity toggles, leaf-location
toggle, one Diagnose button. Nothing else.

Result — top-3 diagnosis with confidence bars, optional
"Show attention" expander that highlights which context dimensions
drove the decision

History — list of past diagnoses with thumbnail and timestamp,
filterable by crop

The context form deliberately has only four questions. Every additional
question is a reason for a farmer to close the app.

Screenshots of the running Gradio MVP and Swagger UI are in
docs/screenshots/.

Deployment plan
Stage	Where	Why
Training	Colab T4	Free, reproducible, enough VRAM for a 0.17 M model
Conversion	PyTorch → ONNX → TFLite	TFLite is what Android gives us for free
MVP web demo	Gradio + HF Spaces	Supervisor review, quick iteration
REST API	FastAPI + Docker on DigitalOcean	Backend for AgroGram
Mobile client	React Native PWA	Runs on any Android browser without an app store
Platform backend	Django REST + PostgreSQL + Redis	AgroGram users, farms, diagnoses, gamification
Field validation	Busoga sub-region	20–30 youth agripreneurs, 12 weeks
The intended production target is a Progressive Web App, not a native
APK. A farmer opens a URL, grants camera permission, uses it. No install,
no update, no Play Store review cycle.

INT8 quantization via TFLite is what brings the model from ~0.65 MB down
to the <0.2 MB target. The notebook shows the size reduction; the
production conversion step is documented in the FastAPI section.

What this is not
It is not a replacement for an extension officer. It is a first-opinion
tool for the case where there isn't one nearby.

It is not trained on field images. PlantVillage is laboratory-condition
images with clean backgrounds. That is the known limitation and the
reason the field validation phase exists.

The context vector in the notebook is synthetic. It is generated with a
weak, documented correlation to the disease class so that attention has
something to learn from. Once field metadata is collected, the generator
is swapped for the real form values. No architecture change needed.

Video demo
demo.mp4 (6 min). Walks through the notebook, the Gradio MVP, the
Swagger UI, and the attention visualization. Focus is on functionality,
not on the research background.

Citation
If you reference this work:

Kakooza, M. (2026). AgriYouthNet-Context+: Cross-Attention Fusion for
Ultra-Lightweight Mobile Plant Disease Diagnosis to Enable Youth
Agricultural Engagement in Eastern Uganda. BSc Software Engineering
capstone, African Leadership University.

License
MIT. Use it, fork it, break it, improve it.

Acknowledgments
PlantVillage (Hughes & Salathé, 2015) for the dataset

Mr Emmanuel Adjei for supervision

The youth agripreneurs in Busoga who gave early feedback on the
context form questions