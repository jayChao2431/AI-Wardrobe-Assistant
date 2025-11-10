"""
AI-Powered Wardrobe Recommender - Streamlit UI
Interactive web application for clothing recommendation system
"""
import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import streamlit as st
import matplotlib.pyplot as plt

from src.model import ResNet50Classifier
from src.recommender import recommend_topk

# Configuration
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "model_best.pth")
CLASSMAP = os.path.join(RESULTS_DIR, "class_to_idx.json")
GALLERY_INDEX = os.path.join(RESULTS_DIR, "gallery_index.npz")
GALLERY_META = os.path.join(RESULTS_DIR, "gallery_meta.json")

# Page configuration
st.set_page_config(
    page_title="AI Wardrobe Recommender",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .recommendation-card {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained ResNet50 model (cached)"""
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(CLASSMAP, 'r') as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    return model, idx_to_class, device


@st.cache_resource
def load_gallery():
    """Load gallery index (cached)"""
    if not os.path.exists(GALLERY_INDEX):
        return None, None

    gallery_data = np.load(GALLERY_INDEX)
    gallery_feats = gallery_data['feats']

    with open(GALLERY_META, 'r') as f:
        gallery_meta = json.load(f)

    return gallery_feats, gallery_meta


def extract_features(image, model, device):
    """Extract features from uploaded image"""
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, features = model(x)
        pred_class = logits.argmax(1).item()
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max().item()

        # Get top-3 predictions
        top3_probs, top3_classes = torch.topk(probs[0], 3)
        top3_predictions = [
            (idx_to_class[str(cls.item())], prob.item())
            for cls, prob in zip(top3_classes, top3_probs)
        ]

    return features.cpu().numpy(), pred_class, confidence, top3_predictions


def main():
    # Header
    st.markdown('<div class="main-header">👔 AI-Powered Wardrobe Recommender</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a clothing item to get personalized style recommendations</div>',
                unsafe_allow_html=True)

    # Load model and gallery
    try:
        model, idx_to_class, device = load_model()
        gallery_feats, gallery_meta = load_gallery()

        if gallery_feats is None:
            st.error(
                "⚠️ Gallery index not found! Please run `python src/build_gallery_index.py` first.")
            st.info("This will create a feature database from your Polyvore dataset.")
            st.stop()

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model info
        st.subheader("📊 Model Information")
        st.info(f"""
        **Architecture:** ResNet50  
        **Classes:** {len(idx_to_class)}  
        **Device:** {device.upper()}  
        **Gallery Size:** {gallery_feats.shape[0]:,} images
        """)

        # Recommendation settings
        st.subheader("🔧 Recommendation Settings")
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            step=1
        )

        # Category filter
        enable_filter = st.checkbox("Enable category filtering", value=False)
        if enable_filter:
            st.info(
                "Filter recommendations by complementary categories (e.g., tops → bottoms)")

        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This AI system uses deep learning to:
        1. Classify clothing items
        2. Extract feature vectors
        3. Find similar items via cosine similarity
        
        **Dataset:** DeepFashion + Polyvore  
        **Categories:** Blazer, Blouse, Dress, Skirt, Tee
        """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a clothing image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of a clothing item"
        )

        # Example images
        if uploaded_file is None:
            st.info(
                "💡 **Tip:** You can use test images from `data/deepfashion_subset/test/` folder")

            # Show example
            example_path = "data/deepfashion_subset/test"
            if os.path.exists(example_path):
                categories = [d for d in os.listdir(
                    example_path) if os.path.isdir(os.path.join(example_path, d))]
                if categories:
                    selected_category = st.selectbox(
                        "Or try an example:", ["None"] + categories)
                    if selected_category != "None":
                        category_path = os.path.join(
                            example_path, selected_category)
                        images = [f for f in os.listdir(
                            category_path) if f.endswith(('.jpg', '.png'))]
                        if images:
                            example_image_path = os.path.join(
                                category_path, images[0])
                            uploaded_file = example_image_path

        # Process uploaded image
        if uploaded_file is not None:
            # Load image
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file).convert('RGB')
            else:
                image = Image.open(uploaded_file).convert('RGB')

            # Display
            st.image(image, caption="Query Image", use_container_width=True)

            # Analyze button
            if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Extract features
                    query_feats, pred_class, confidence, top3_preds = extract_features(
                        image, model, device)

                    # Display predictions
                    st.success("✅ Analysis Complete!")

                    st.subheader("📊 Classification Results")

                    # Main prediction
                    pred_label = idx_to_class[str(pred_class)]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Predicted Category", pred_label)
                    with col_b:
                        st.metric("Confidence", f"{confidence*100:.1f}%")

                    # Top-3 predictions
                    with st.expander("View all predictions"):
                        for i, (label, prob) in enumerate(top3_preds, 1):
                            st.write(f"{i}. **{label}**: {prob*100:.2f}%")
                            st.progress(prob)

                    # Store in session state
                    st.session_state.query_feats = query_feats
                    st.session_state.pred_label = pred_label
                    st.session_state.confidence = confidence

    with col2:
        st.header("🎯 Recommendations")

        if 'query_feats' in st.session_state:
            with st.spinner(f"Finding top {num_recommendations} similar items..."):
                # Get recommendations
                recommendations = recommend_topk(
                    st.session_state.query_feats,
                    gallery_feats,
                    gallery_meta,
                    k=num_recommendations
                )

                st.success(f"Found {len(recommendations)} recommendations!")

                # Display recommendations
                for i, (meta, similarity) in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(
                            f"### Rank {i} - Similarity: {similarity:.3f}")

                        rec_path = meta['image_path']

                        # Check if file exists
                        if os.path.exists(rec_path):
                            rec_img = Image.open(rec_path).convert('RGB')

                            col_img, col_info = st.columns([1, 1])

                            with col_img:
                                st.image(rec_img, use_container_width=True)

                            with col_info:
                                st.write(
                                    f"**File:** {os.path.basename(rec_path)}")
                                st.write(
                                    f"**Similarity Score:** {similarity:.4f}")

                                # Similarity bar
                                st.progress(similarity)

                                # Match quality
                                if similarity > 0.9:
                                    st.success("🟢 Excellent Match")
                                elif similarity > 0.8:
                                    st.info("🔵 Good Match")
                                elif similarity > 0.7:
                                    st.warning("🟡 Fair Match")
                                else:
                                    st.error("🔴 Weak Match")
                        else:
                            st.warning(f"Image not found: {rec_path}")

                        st.divider()
        else:
            st.info(
                "👆 Upload an image and click 'Analyze & Recommend' to see recommendations")

            # Show placeholder
            st.image("https://via.placeholder.com/400x300.png?text=Your+Recommendations+Will+Appear+Here",
                     use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 AI-Powered Wardrobe Recommender | Built with ResNet50 + Streamlit</p>
        <p>Dataset: DeepFashion + Polyvore | Model: Transfer Learning with ImageNet</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
