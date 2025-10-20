import streamlit as st
from PIL import Image
from src.recommender import suggest_outfit

st.set_page_config(page_title="AI Wardrobe Assistant", layout="centered")
st.title("👗 AI-Powered Wardrobe Assistant")
st.write("Upload a clothing image to preview a demo suggestion.")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

    # Placeholder predictions for Deliverable 1
    st.success("Sample classification: Top (Blue)")
    st.info(suggest_outfit("Top", "Blue"))
else:
    st.caption("Tip: Use any of the sample images in data/deepfashion_samples.")
