# src/gradio_app.py
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from model import AgriYouthNetContextPlus, CLASS_NAMES

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'D:\\agriyouthnet-contextplus\\results\\models\\agriyouth_contextplus.pt'

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

model = AgriYouthNetContextPlus(fusion='attention', n_heads=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict(img, crop_label, wet, severe, on_stem):
    if img is None:
        return {}
    crop_map = {'Maize': 0.0, 'Tomato': 0.5, 'Potato': 1.0}
    ctx = torch.tensor(
        [[crop_map[crop_label], float(wet), float(severe), float(on_stem)]],
        dtype=torch.float32).to(DEVICE)
    x = eval_tf(img.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x, ctx), 1)[0].cpu().numpy()
    return {CLASS_NAMES[i].replace('_', ' '): float(probs[i])
            for i in range(len(CLASS_NAMES))}


if __name__ == '__main__':
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type='pil', label='Leaf photo'),
            gr.Radio(['Maize', 'Tomato', 'Potato'], label='Crop', value='Tomato'),
            gr.Radio([0, 1], label='Wet season?', value=1),
            gr.Radio([0, 1], label='Severe symptoms?', value=1),
            gr.Radio([0, 1], label='Symptoms on stem?', value=0),
        ],
        outputs=gr.Label(num_top_classes=3, label='Diagnosis'),
        title='AgriYouthNet-Context+',
    )
    demo.launch()