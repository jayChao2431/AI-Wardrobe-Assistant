# All comments in English.
import json, os
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import ResNet50Classifier
from src.recommender import recommend_topk

st.set_page_config(page_title="AI Wardrobe Recommender", page_icon="👗", layout="centered")

RESULTS_DIR = "results"
MODEL_PATH  = os.path.join(RESULTS_DIR, "model_best.pth")
CLASSMAP    = os.path.join(RESULTS_DIR, "class_to_idx.json")
GALLERY_NPZ = os.path.join(RESULTS_DIR, "gallery_index.npz")
GALLERY_META= os.path.join(RESULTS_DIR, "gallery_meta.json")

st.title("👗 AI-Powered Wardrobe Recommender (ResNet50)")
st.caption("Upload a clothing image → predicted class and recommendations.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

device = "cuda" if torch.cuda.is_available() else "cpu"
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSMAP)):
        st.error("Trained model not found. Run training first (results/model_best.pth).")
        st.stop()

    with open(CLASSMAP, 'r') as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, feats = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = idx_to_class[str(pred_idx)] if isinstance(idx_to_class, dict) else idx_to_class[pred_idx]

    st.success(f"Predicted class: {pred_label}")

    if os.path.exists(GALLERY_NPZ) and os.path.exists(GALLERY_META):
        data = np.load(GALLERY_NPZ)
        gallery_feats = data["feats"]
        with open(GALLERY_META, 'r') as f:
            meta = json.load(f)
        recs = recommend_topk(feats.squeeze(0).cpu().numpy(), gallery_feats, meta, k=5)
        st.subheader("Recommended items:")
        cols = st.columns(5)
        for i,(m,score) in enumerate(recs):
            with cols[i % 5]:
                st.image(m["image_path"], caption=f"sim={score:.2f}", use_column_width=True)
    else:
        st.info("No gallery index found. Run: python src/build_gallery_index.py")
