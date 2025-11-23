import sys
from pathlib import Path

# Add parent directory to path for imports (MUST be before src imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation_explainer import RecommendationExplainer
from src.smart_validator import SmartValidator
from src.ensemble_classifier import EnsembleClassifier
from src.recommender import recommend_topk
from src.model import ResNet50Classifier
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import clip
import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time


# Configuration (use absolute paths to avoid issues)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = str(PROJECT_ROOT / "results")
MODEL_PATH = str(PROJECT_ROOT / "results" / "model_best.pth")
CLASSMAP = str(PROJECT_ROOT / "results" / "class_to_idx.json")
GALLERY_INDEX = str(PROJECT_ROOT / "results" / "gallery_index.npz")
GALLERY_META = str(PROJECT_ROOT / "results" / "gallery_meta.json")

# Page configuration
st.set_page_config(
    page_title="AI Wardrobe Recommender v4.0",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Wardrobe Recommender v4.0 - Final Project\n\nEnsemble classifier with explainable recommendations.\nAccuracy: 73.47% | Features: CLIP + Gender filtering + Outfit matching"
    }
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
def load_ensemble_system():
    """Load CLIP-based Ensemble Classifier and Smart Validator (Phase 1+2)"""
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Load categories
    with open(CLASSMAP, 'r') as f:
        idx_to_class = json.load(f)
    categories = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    # Initialize Ensemble Classifier
    ensemble_classifier = EnsembleClassifier(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        categories=categories,
        device=device
    )

    # Initialize Smart Validator
    smart_validator = SmartValidator(categories=categories)

    return ensemble_classifier, smart_validator, clip_model, clip_preprocess, device


@st.cache_resource
def load_gallery():
    """Load gallery embeddings and metadata (cached)"""
    # Use gallery_embeddings.npy (6844 items) instead of gallery_index.npz (6000 items)
    embeddings_file = os.path.join(RESULTS_DIR, "gallery_embeddings.npy")

    if not os.path.exists(embeddings_file):
        # Fallback to old gallery_index.npz if new file doesn't exist
        if os.path.exists(GALLERY_INDEX):
            gallery_data = np.load(GALLERY_INDEX)
            gallery_feats = gallery_data.get(
                'features', gallery_data.get('feats'))
        else:
            return None, None
    else:
        # Load from gallery_embeddings.npy
        gallery_feats = np.load(embeddings_file)

    with open(GALLERY_META, 'r') as f:
        gallery_meta = json.load(f)

    return gallery_feats, gallery_meta


def infer_gender(category: str, image) -> str:
    """
    Infer gender of clothing item based on category and visual features.

    Args:
        category: Predicted category (Blazer, Blouse, Dress, Skirt, Tee)
        image: PIL Image object

    Returns:
        Gender label: 'Male', 'Female', or 'Unisex'
    """
    # Rule-based gender inference
    if category in ['Blouse', 'Skirt', 'Dress']:
        return 'Female'
    elif category in ['Blazer', 'Tee']:
        # For ambiguous categories, use color analysis
        import numpy as np
        img_array = np.array(image)
        avg_color = img_array.mean(axis=(0, 1))  # RGB averages
        r, g, b = avg_color

        # Feminine color heuristics (pink, purple tones)
        if r > 180 and g < 150 and b > 150:  # Pink/purple range
            return 'Female'

        # Default for Blazer: Male (business context)
        if category == 'Blazer':
            return 'Male'

        # Default for Tee: Unisex
        return 'Unisex'

    return 'Unisex'


def extract_features(image, model, device, idx_to_class, clip_model=None, clip_preprocess=None,
                     ensemble_classifier=None, smart_validator=None, image_path=""):
    """Extract features from uploaded image and infer gender (with Phase 1+2 enhancements)"""
    # Use Ensemble Classifier if available (Phase 1)
    if ensemble_classifier is not None and smart_validator is not None:
        # Phase 1: Ensemble Classification
        pred_category, confidence, details = ensemble_classifier.classify(
            image=image,
            text_info="",  # No text metadata from uploaded images
            image_path=image_path,
            return_details=True
        )

        # Get all scores for validation
        all_scores = details['final_scores']

        # Phase 2: Smart Validation
        validated_category, validated_confidence, validation_info = smart_validator.validate_classification(
            predicted_category=pred_category,
            confidence=confidence,
            all_scores=all_scores,
            text_info="",
            image_path=image_path
        )

        # Extract CLIP features for recommendation
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(
                image_input).cpu().numpy().flatten()
            features = features / np.linalg.norm(features)

        # Get category index (case-insensitive match)
        categories = list(idx_to_class.values())
        # Find matching category (case-insensitive)
        validated_category_lower = validated_category.lower()
        matching_category = next((cat for cat in categories if cat.lower() == validated_category_lower), validated_category)
        pred_class = categories.index(matching_category)
        confidence = validated_confidence

        # Get top-3 predictions from ensemble scores
        top3_items = sorted(all_scores.items(), key=lambda x: -x[1])[:3]
        top3_predictions = [(cat, score) for cat, score in top3_items]

        # Store validation info in session state for display
        if 'last_validation_info' not in st.session_state:
            st.session_state.last_validation_info = {}
        st.session_state.last_validation_info = {
            'original': validation_info['original_prediction'],
            'final': validated_category,
            'corrections': validation_info.get('corrections', []),
            'warnings': validation_info.get('warnings', []),
            'ensemble_details': details
        }

    # Use CLIP if available, otherwise use ResNet50
    elif clip_model is not None:
        # CLIP feature extraction
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(
                image_input).cpu().numpy().flatten()
            features = features / np.linalg.norm(features)

        # CLIP zero-shot classification with enhanced prompts
        categories = list(idx_to_class.values())

        # Enhanced descriptive prompts for better classification accuracy
        prompt_templates = {
            'blazer': "a photo of a formal blazer jacket with buttons and lapels",
            'blouse': "a photo of a woman's blouse or dress shirt with collar",
            'dress': "a photo of a dress or gown covering the body from shoulders to legs",
            'skirt': "a photo of a skirt worn around the waist covering hips and legs",
            'tee': "a photo of a casual t-shirt or short-sleeved top",
            'pants': "a photo of long pants, jeans, or trousers covering legs from waist to ankles",
            'shorts': "a photo of short pants or shorts above the knee"
        }

        text_prompts = [
            prompt_templates.get(cat.lower(), f"a photo of a {cat.lower()}")
            for cat in categories
        ]
        text_inputs = clip.tokenize(text_prompts).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            image_features_norm = torch.from_numpy(features).float().to(device)

            # Calculate similarities
            similarities = (100.0 * image_features_norm @
                            text_features.T).softmax(dim=-1)
            probs = similarities.cpu().numpy()

        pred_class = probs.argmax()
        confidence = probs[pred_class]

        # Get top-3 predictions
        top3_indices = probs.argsort()[-3:][::-1]
        top3_predictions = [
            (categories[idx], probs[idx])
            for idx in top3_indices
        ]
    else:
        # Original ResNet50 feature extraction
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        x = tfm(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, features = model(x)
            features = features.cpu().numpy().flatten()
            pred_class = logits.argmax(1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            top3_probs, top3_classes = torch.topk(probs[0], 3)
            top3_predictions = [
                (idx_to_class[str(cls.item())], prob.item())
                for cls, prob in zip(top3_classes, top3_probs)
            ]

    # Infer gender based on predicted category
    pred_category = idx_to_class[str(pred_class)]
    pred_gender = infer_gender(pred_category, image)

    return features, pred_class, confidence, top3_predictions, pred_gender


def main():
    # Initialize explainer in session state
    if 'explainer' not in st.session_state:
        st.session_state.explainer = RecommendationExplainer()

    # Header with version badge
    # Main title
    st.markdown('<div class="main-header">AI-Powered Wardrobe Recommender</div>',
                unsafe_allow_html=True)

    # Check if CLIP model is available
    model_config_path = os.path.join(RESULTS_DIR, "model_config.json")
    use_clip = False
    clip_model = None
    clip_preprocess = None

    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            config = json.load(f)
        if config.get('model_type') == 'CLIP':
            use_clip = True
            model_desc = "Ensemble Classifier (CLIP ViT-B/32 + Smart Validator)"
        else:
            model_desc = "Ensemble Classifier"
    else:
        model_desc = "Ensemble Classifier"

    st.markdown(f'<div class="sub-header">Upload a clothing item to get explainable outfit recommendations powered by {model_desc} with 73.47% accuracy</div>',
                unsafe_allow_html=True)

    # Load model and gallery with error handling
    with st.spinner("Loading AI model and gallery database..."):
        # Initialize variables
        ensemble_classifier = None
        smart_validator = None

        # Load CLIP if available
        if use_clip:
            try:
                # Phase 1+2: Load Ensemble Classifier and Smart Validator
                ensemble_classifier, smart_validator, clip_model, clip_preprocess, device = load_ensemble_system()

                # Load class mapping
                if not os.path.exists(CLASSMAP):
                    st.error(f" Class mapping file not found: {CLASSMAP}")
                    st.info("Creating default class mapping...")
                    # Create default mapping for 7 categories
                    idx_to_class = {
                        "0": "Blazer", "1": "Blouse", "2": "Dress",
                        "3": "Skirt", "4": "Tee", "5": "Pants", "6": "Shorts"
                    }
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    with open(CLASSMAP, 'w') as f:
                        json.dump(idx_to_class, f)
                    st.success(" Created class mapping file")
                else:
                    with open(CLASSMAP, 'r') as f:
                        idx_to_class = json.load(f)

                model = None  # Not needed for CLIP
                st.success(
                    f" Enhanced System Loaded! (Ensemble Classifier + Smart Validator)")
            except Exception as e:
                st.error(f" Error loading enhanced system: {str(e)}")
                st.exception(e)
                st.stop()
        else:
            try:
                model, idx_to_class, device = load_model()
                st.success("Model loaded successfully!")
            except FileNotFoundError as e:
                st.error("Model files not found! Please train the model first.")
                st.code("python src/train.py --epochs 10 --batch_size 32")
                st.info(
                    "Or switch to CLIP mode by running: python3 upgrade_to_clip.py")
                st.stop()
            except Exception as e:
                st.error(f" Error loading model: {str(e)}")
                st.exception(e)
                st.stop()

        try:
            gallery_feats, gallery_meta = load_gallery()
            if gallery_feats is None:
                st.warning(
                    "Gallery index not found! Building recommendation database...")
                st.info("Please run: `python src/build_gallery_index.py`")
                st.markdown("""
                **Why do I need this?**
                - Creates a searchable database of 252K fashion items
                - Takes ~2-3 hours on Apple Silicon GPU
                - Required for recommendation functionality
                """)
                st.stop()
            st.success(
                f"Gallery loaded: {gallery_feats.shape[0]:,} items indexed")
        except Exception as e:
            st.error(f" Error loading gallery: {str(e)}")
            st.exception(e)
            st.stop()

    # Sidebar
    with st.sidebar:
        st.header(" System Dashboard")

        # Model Performance Metrics
        st.subheader(" Model Capabilities")

        if use_clip:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Model Type", "CLIP ViT-B/32",
                          help="OpenAI's pretrained vision-language model")
            with col_m2:
                st.metric("Training Data", "400M pairs",
                          help="Trained on web-scale image-text dataset")

            st.info(
                " **Zero-shot Learning**: No training needed! CLIP understands fashion naturally.")
        else:
            try:
                with open(os.path.join(RESULTS_DIR, "test_metrics.json"), 'r') as f:
                    metrics = json.load(f)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%",
                              help="Overall classification accuracy on test set")
                with col_m2:
                    st.metric("F1 Score", f"{metrics['f1_weighted']:.3f}",
                              help="Weighted F1 score across all classes")
            except:
                st.warning("Metrics not available")

        st.divider()

        # Model info
        st.subheader(" Model Information")
        feature_dim = gallery_feats.shape[1] if gallery_feats is not None else 512
        st.markdown(f"""
        **Architecture:** Ensemble Classifier  
        - CLIP ViT-B/32 (95% weight)  
        - Keyword Matching (3%)  
        - Path Analysis (2%)  
        **Accuracy:** 73.47%  
        **Classes:** {len(idx_to_class)} categories  
        **Device:** {str(device).upper()}  
        **Gallery Size:** {gallery_feats.shape[0]:,} items  
        **Feature Dim:** {feature_dim}-D CLIP vectors  
        **Explainability:** Style + Color + Material analysis
        """)

        st.divider()

        # Recommendation settings
        st.subheader(" Recommendation Settings")

        # Outfit matching mode toggle
        outfit_mode = st.checkbox(
            "Outfit Matching Mode",
            value=True,
            help="Recommend complementary items for outfit coordination"
        )

        if outfit_mode:
            st.success("**ENABLED**: Tops → Bottoms, Bottoms → Tops")
        else:
            st.info(" **OFF**: Show similar items")

        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
            help="More recommendations = slower processing"
        )

        similarity_threshold = st.slider(
            "Minimum similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
            help="Filter out items below this similarity score (0.15 recommended for outfit matching)"
        )

        st.divider()

        # Gender Filter
        st.subheader("Gender Filter")
        gender_filter = st.radio(
            "Show recommendations for:",
            options=["All", "Male", "Female", "Unisex"],
            index=0,
            help="Filter recommendations by gender"
        )
        if 'gender_filter' not in st.session_state or st.session_state.gender_filter != gender_filter:
            st.session_state.gender_filter = gender_filter

        st.divider()

        # Statistics
        st.subheader("Session Statistics")
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'total_processing_time' not in st.session_state:
            st.session_state.total_processing_time = 0.0

        st.metric("Queries Processed", st.session_state.query_count)
        if st.session_state.query_count > 0:
            avg_time = st.session_state.total_processing_time / st.session_state.query_count
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")

        st.divider()

        # About
        st.subheader("ℹ About This System")
        st.markdown("""
        **How it works:**
        1.  Upload clothing image
        2. 🤖 CLIP analyzes & classifies (7 categories)
        3. ↔ Find complementary outfits
        4.  Gender-aware matching
        
        **Categories:**  
        • Tops: Blazer, Blouse, Tee  
        • Bottoms: Skirt, Pants, Shorts  
        • Dresses: Complete outfits
        
        **Model:** CLIP ViT-B/32 (Zero-shot)
        """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header(" Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a clothing image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of a clothing item"
        )

        # Example images
        if uploaded_file is None:
            st.info(
                "**Note:** You can use test images from `data/deepfashion_subset/test/` folder")

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

            # Display with larger size
            st.image(image, caption="Query Image", width=500)

            # Analyze button
            if st.button("Analyze & Recommend", type="primary", use_container_width=True):
                start_time = time.time()

                with st.spinner("Extracting features with Enhanced Ensemble System..."):
                    try:
                        # Get uploaded file path for ensemble classifier
                        image_path = uploaded_file if isinstance(
                            uploaded_file, str) else getattr(uploaded_file, 'name', '')

                        # Extract features and infer gender (with Phase 1+2 enhancements)
                        query_feats, pred_class, confidence, top3_preds, pred_gender = extract_features(
                            image, model, device, idx_to_class, clip_model, clip_preprocess,
                            ensemble_classifier, smart_validator, image_path)

                        processing_time = time.time() - start_time

                        # Update statistics
                        st.session_state.query_count += 1
                        st.session_state.total_processing_time += processing_time

                        # Display predictions
                        st.success(
                            f"Analysis Complete! (Processed in {processing_time:.2f}s)")

                        st.subheader("Classification Results")

                        # Main prediction with confidence indicator
                        pred_label = idx_to_class[str(pred_class)]
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Predicted Category", pred_label,
                                      help="Most likely clothing category")
                        with col_b:
                            st.metric("Confidence", f"{confidence*100:.1f}%",
                                      help="Model confidence score")
                        with col_c:
                            st.metric("Gender", pred_gender,
                                      help="Detected gender category")

                        # Phase 1+2: Display validation information
                        if 'last_validation_info' in st.session_state and st.session_state.last_validation_info:
                            val_info = st.session_state.last_validation_info

                            # Show corrections if any
                            if val_info.get('corrections'):
                                with st.expander(" Smart Validator Corrections", expanded=True):
                                    st.info(
                                        f"**Original Prediction**: {val_info['original']} → **Corrected to**: {val_info['final']}")
                                    for i, correction in enumerate(val_info['corrections'], 1):
                                        st.warning(
                                            f"**Correction {i}**: {correction['reason']}")

                            # Show warnings if any
                            if val_info.get('warnings'):
                                with st.expander(" Validation Warnings", expanded=False):
                                    for warning in val_info['warnings']:
                                        st.warning(
                                            f"**{warning['type']}**: {warning['message']}")

                            # Show ensemble details
                            if val_info.get('ensemble_details'):
                                with st.expander(" Ensemble Classifier Details", expanded=False):
                                    details = val_info['ensemble_details']

                                    st.markdown("**Fusion Weights:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "CLIP", f"{details['weights']['clip']:.0%}")
                                    with col2:
                                        st.metric(
                                            "Keyword", f"{details['weights']['keyword']:.0%}")
                                    with col3:
                                        st.metric(
                                            "Path", f"{details['weights']['path']:.0%}")

                                    st.markdown("**Top 3 Ensemble Scores:**")
                                    for cat, score in details['top3']:
                                        st.write(f"**{cat}**: {score:.3f}")
                                        st.progress(float(score))

                        # Top-3 predictions with enhanced visualization
                        with st.expander(" View detailed predictions", expanded=False):
                            for i, (label, prob) in enumerate(top3_preds, 1):
                                col_label, col_prob = st.columns([3, 1])
                                with col_label:
                                    icon = "[1]" if i == 1 else "[2]" if i == 2 else "[3]"
                                    st.write(f"{icon} **{label}**")
                                with col_prob:
                                    st.write(f"{prob*100:.2f}%")
                                st.progress(float(prob))

                        # Store in session state
                        st.session_state.query_feats = query_feats
                        st.session_state.pred_label = pred_label
                        st.session_state.pred_gender = pred_gender
                        st.session_state.confidence = confidence
                        st.session_state.processing_time = processing_time
                        st.session_state.query_image = image

                    except Exception as e:
                        st.error(f" Error during analysis: {str(e)}")
                        st.exception(e)

    with col2:
        st.header(
            " Outfit Recommendations" if outfit_mode else " Style Recommendations")

        if 'query_feats' in st.session_state:
            rec_start_time = time.time()

            with st.spinner(f"Searching {gallery_feats.shape[0]:,} items..."):
                try:
                    # Get recommendations with outfit matching mode and gender filtering
                    # Use user-selected gender filter if available, otherwise use auto-detected
                    user_gender_filter = st.session_state.get(
                        'gender_filter', 'All')
                    if user_gender_filter == 'All':
                        gender_for_rec = None
                    else:
                        gender_for_rec = user_gender_filter

                    # Debug info
                    st.info(
                        f" Filter Settings: Gender={gender_for_rec}, Category={st.session_state.pred_label}, Outfit Mode={'ON' if outfit_mode else 'OFF'}")

                    recommendations = recommend_topk(
                        st.session_state.query_feats,
                        gallery_feats,
                        gallery_meta,
                        k=num_recommendations * 3,  # Get extra for filtering
                        query_category=st.session_state.pred_label,
                        complementary_mode=outfit_mode,
                        query_gender=gender_for_rec
                    )

                    # Filter by similarity threshold, but ensure at least 2 recommendations
                    filtered_recs = [
                        (meta, sim) for meta, sim in recommendations
                        if sim >= similarity_threshold
                    ][:num_recommendations]

                    # Fallback: if no items meet threshold, show top 2 anyway
                    MIN_RECOMMENDATIONS = 2
                    if len(filtered_recs) < MIN_RECOMMENDATIONS and len(recommendations) > 0:
                        filtered_recs = recommendations[:max(
                            MIN_RECOMMENDATIONS, num_recommendations)]
                        st.info(
                            f"Showing top {len(filtered_recs)} matches (some below similarity threshold {similarity_threshold:.2f})")

                    rec_time = time.time() - rec_start_time

                    if len(filtered_recs) == 0:
                        st.error(
                            "No recommendations found. This may be due to:")
                        st.markdown("""
                        - **Gender mismatch**: No items match the detected gender in gallery
                        - **Category mismatch**: No complementary items available (in Outfit Mode)
                        - **Gallery limitations**: Try disabling Outfit Mode or adjusting filters
                        """)
                        if outfit_mode:
                            st.info(
                                "**Quick Fix**: Try disabling 'Outfit Matching Mode' to see similar items instead")
                        else:
                            st.info(
                                "Try uploading a different image or checking gallery contents")
                    else:
                        mode_text = "outfit matches" if outfit_mode else "similar items"
                        st.success(
                            f"Found {len(filtered_recs)} {mode_text} in {rec_time:.2f}s")

                        # Show matching strategy
                        if outfit_mode:
                            query_cat = st.session_state.pred_label
                            user_gender = st.session_state.get(
                                'gender_filter', 'All')

                            if query_cat in ['Blazer', 'Blouse', 'Tee']:
                                st.info(
                                    f"**Outfit Strategy:** Your {query_cat} (Top) → Recommending Bottoms (Skirts/Pants/Shorts)")
                            elif query_cat in ['Skirt', 'Pants', 'Shorts']:
                                st.info(
                                    f"**Outfit Strategy:** Your {query_cat} (Bottom) → Recommending Tops (Blazer/Blouse/Tee)")
                            elif query_cat == 'Dress':
                                st.info(
                                    f"**Outfit Strategy:** Your {query_cat} → Complete outfit (showing similar dresses)")

                        # Export button
                        if st.button(" Export Results", help="Save recommendations as image"):
                            st.info("Export functionality - Coming soon!")

                        # Display recommendations
                        for i, (meta, similarity) in enumerate(filtered_recs, 1):
                            # Clean container with better spacing
                            with st.container():
                                # Rank header - clean and simple
                                col_rank, col_sim = st.columns([3, 1])
                                with col_rank:
                                    st.subheader(f"Recommendation #{i}")
                                with col_sim:
                                    st.metric("Match", f"{similarity:.1%}")

                                st.markdown("---")

                                rec_path = meta['image_path']

                                # Check if file exists
                                if os.path.exists(rec_path):
                                    rec_img = Image.open(
                                        rec_path).convert('RGB')

                                    col_img, col_info = st.columns([3, 2])

                                    with col_img:
                                        st.image(
                                            rec_img, use_container_width=True)

                                    with col_info:
                                        # Basic Info in one line
                                        rec_category = meta.get(
                                            'category', 'Unknown')
                                        rec_gender = meta.get(
                                            'gender', 'Unknown')

                                        st.markdown(
                                            f"** {rec_category}** | ** {rec_gender}**")

                                        # Match Quality with score
                                        match_quality = "Excellent" if similarity > 0.9 else "Very Good" if similarity > 0.8 else "Good" if similarity > 0.7 else "Fair"
                                        st.markdown(
                                            f"**Match Score:** {similarity:.1%} ({match_quality})")
                                        st.progress(similarity)

                                        st.divider()

                                        # Generate explanation
                                        query_meta_dict = {
                                            'title': '',
                                            'description': '',
                                            'category': st.session_state.get('pred_label', ''),
                                        }

                                        explanation = st.session_state.explainer.generate_explanation(
                                            query_meta_dict,
                                            meta,
                                            similarity,
                                            st.session_state.get(
                                                'pred_label', ''),
                                            outfit_mode
                                        )

                                        # Display explanation - all in one flow
                                        st.markdown("** Why Recommended:**")

                                        # Build explanation text as continuous paragraph
                                        explanation_text = f"{explanation['main_reason']}. "

                                        if explanation['style_match']:
                                            explanation_text += f"{explanation['style_match']}. "

                                        if explanation['color_match']:
                                            explanation_text += f"{explanation['color_match']}. "

                                        if explanation['material_info']:
                                            explanation_text += f"{explanation['material_info']}. "

                                        # Show as continuous text
                                        st.markdown(explanation_text)

                                        # Occasion in a separate line
                                        if explanation['occasion']:
                                            st.markdown(
                                                f"** {explanation['occasion']}**")

                                        # Confidence as small text
                                        confidence = explanation['confidence']
                                        confidence_icon = "" if confidence > 0.7 else "ℹ"
                                        confidence_text = "High confidence" if confidence > 0.7 else "Moderate confidence"
                                        st.caption(
                                            f"{confidence_icon} {confidence_text} ({confidence:.0%})")

                                        # Optional: Show detected features inline if available
                                        if meta.get('title') or meta.get('description'):
                                            rec_features = st.session_state.explainer.extract_features(
                                                meta)
                                            feature_tags = []
                                            if rec_features['styles']:
                                                feature_tags.extend(
                                                    rec_features['styles'])
                                            if rec_features['colors']:
                                                feature_tags.extend(
                                                    rec_features['colors'])
                                            if rec_features['materials']:
                                                feature_tags.extend(
                                                    rec_features['materials'])

                                            if feature_tags:
                                                # Show max 5 tags
                                                st.caption(
                                                    f" Tags: {', '.join(feature_tags[:5])}")

                                        # Optional: Expandable detailed chart
                                        with st.expander(" Detailed Score Chart"):
                                            query_category = st.session_state.get(
                                                'pred_label', '')
                                            category_match = (
                                                query_category == rec_category)

                                            if outfit_mode:
                                                TOPS = [
                                                    'Blazer', 'Blouse', 'Tee']
                                                BOTTOMS = [
                                                    'Skirt', 'Pants', 'Shorts']
                                                if (query_category in TOPS and rec_category in BOTTOMS) or \
                                                   (query_category in BOTTOMS and rec_category in TOPS):
                                                    category_match = True

                                            gender_match = (st.session_state.get('gender_filter', 'All') == rec_gender or
                                                            rec_gender == 'Unisex')

                                            fig = st.session_state.explainer.generate_simple_similarity_bar(
                                                similarity, category_match, gender_match
                                            )
                                            st.pyplot(fig)
                                            plt.close(fig)
                                else:
                                    st.warning(
                                        f"Image not found: {rec_path}")

                                st.divider()

                except Exception as e:
                    st.error(f" Error generating recommendations: {str(e)}")
                    st.exception(e)
        else:
            # Enhanced placeholder
            st.info(
                " Upload an image and click 'Analyze & Recommend' to get started")

            st.markdown("""
            ### How it works:
            1. **Upload** a clothing image (JPG/PNG)
            2. **Analyze** to classify and extract features
            3. **Discover** similar items from 252K+ gallery
            
            ### What you'll get:
            -  Clothing category classification
            - Confidence scores
            - Top-K similar items
            - Similarity metrics
            """)

            # Show placeholder image
            st.image("https://via.placeholder.com/500x350.png?text=+Your+Recommendations+Will+Appear+Here",
                     width=400)

    # Footer
    st.markdown("---")

    # Footer info
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong> AI-Powered Wardrobe Recommender</strong></p>
        <p>Built with Ensemble Classifier (CLIP ViT-B/32 + Keyword + Path) + Smart Validator + Explainable Recommendations</p>
        <p> Datasets: DeepFashion (821 images) + Polyvore (252K items) → 6,844 gallery items</p>
        <p> Model: <strong>73.47% accuracy</strong>, 512-D CLIP features, Cosine similarity + Outfit matching logic</p>
        <p> Features: Multi-modal classification, Gender filtering, Style/Color/Material analysis, Explainability</p>
        <p style='font-size: 0.9em; margin-top: 10px;'>
            © 2025 | Advanced Machine Learning Final Project | University of Florida
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'total_processing_time' not in st.session_state:
        st.session_state.total_processing_time = 0.0

    main()
