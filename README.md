# AI-Powered Wardrobe Recommender

An intelligent fashion recommendation system powered by **OpenAI CLIP** with ensemble classification, smart validation, and explainable recommendations.
# AI-Powered Wardrobe Recommender 👔

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![CLIP](https://img.shields.io/badge/CLIP-ViT-B/32-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

An intelligent fashion recommendation system powered by **OpenAI CLIP** with ensemble classification, smart validation, and explainable recommendations.

**Performance:** 73.47% Accuracy | 6,844 Gallery Items | ~0.8s Processing Time
**Architecture:** CLIP ViT-B/32 + Ensemble (95%/3%/2%) + Smart Validator + Recommendation Explainer

---

## Abstract

This project evolved from a ResNet50-based classifier to a CLIP-powered ensemble due to a critical limitation: the original dataset lacked gender labels, making it impossible for ResNet50 to distinguish men's and women's clothing. By leveraging CLIP's zero-shot capabilities and adding ensemble classification with smart validation and automatic gender inference, the system achieved significant improvements:

- Deliverable 2 (ResNet50): 56.6% accuracy
- Deliverable 3 v1 (CLIP Only): 62.0% accuracy
- Deliverable 3 v2 (CLIP + Keyword): 68.0% accuracy
- Deliverable 3 v4 (Full Ensemble): 73.47% accuracy

The current system combines CLIP-based visual understanding, ensemble classification, smart validation, and explainable recommendations. It achieves 73.47% accuracy on fashion categorization and provides detailed explanations for each recommendation.

Key Features:

- Ensemble Classification: CLIP (95%) + Keyword (3%) + Path (2%) fusion
- Smart Validation: Confidence-based error detection and correction
- Explainable Recommendations: Style, color, and occasion matching analysis
- Gender Intelligence: Automatic detection without labeled data
- 6,844 Gallery Items: Curated from Polyvore and DressCode datasets
- Real-time Processing: ~0.8s per image on Apple Silicon

---

## Features

Core Capabilities:

- Zero-Shot Classification: CLIP ViT-B/32 for 7-category recognition (Blazer, Blouse, Dress, Skirt, Tee, Pants, Shorts)
- Ensemble Intelligence: Multi-signal fusion (CLIP 95% + Keyword 3% + Path 2%)
- Smart Validation: Confidence-based error detection with category-specific correction rules
- Explainable AI: Detailed reasoning for each recommendation (style, color, material, occasion)
- Gender Filtering: Automatic gender detection with category-specific rules
- Visual Search: Cosine similarity across 6,844 pre-indexed fashion items
- Web Interface: Streamlit UI with real-time feedback

Latest Performance (Deliverable 3):

- Accuracy: 73.47% on test set (98 images)
- Gallery Size: 6,844 items with CLIP embeddings
- Processing Speed: ~0.8s per image (Apple Silicon MPS)
- Categories: Blazer (56), Blouse (385), Tee (1,633), Dress (2,083), Skirt (1,012), Pants (616), Shorts (215)

---

## System Architecture

User Input Layer: Image upload, file browser, camera capture
Preprocessing: Resize to 224x224, normalization, tensor conversion
CLIP ViT-B/32 Feature Extraction: Zero-shot classification (512-D embeddings), visual-semantic understanding
Ensemble Classifier: CLIP (95%), Keyword (3%), Path (2%) fusion
Smart Validator: Confidence thresholds, error pattern detection
Recommendation Explainer: Style, color, material, occasion, gender-based filtering
Presentation Layer: Streamlit Web UI, session statistics, interactive similarity filtering
Gallery Database: 6,844 items × 512-D CLIP embeddings

---

## Dataset Specifications

### Primary Training Dataset: Polyvore + DressCode (Current System)

- **Source**: Multiple Kaggle datasets integrated
  - [Polyvore Dataset](https://www.kaggle.com/datasets/xthan/polyvore-dataset) - Fashion outfit combinations
  - [DressCode Dataset](https://www.kaggle.com/datasets) - Men's clothing items
  - Additional curated fashion images
- **Total Size**: 6,844 curated fashion items across 7 categories
- **Categories**:
  - **Tops**: Blazer (56, 0.9%), Blouse (385, 6.4%), Tee (1,633, 27.2%)
  - **Bottoms**: Pants (616, 10.3%), Shorts (215, 3.6%)
  - **Dresses**: Dress (2,083, 34.7%), Skirt (1,012, 16.9%)
- **Purpose**:
  - Gallery for recommendation system (CLIP-based visual search)
  - Zero-shot classification using CLIP ViT-B/32
  - Ensemble classification with keyword/path analysis
- **Preprocessing**: Automated category correction, metadata extraction, smart labeling
- **Index Format**:
  - Features: NumPy compressed array (6844 × 512 float32, CLIP embeddings)
  - Metadata: JSON with titles, descriptions, categories, paths

### Legacy Training Dataset: DeepFashion Subset (Reference Only)

- **Source**: [DeepFashion: Category and Attribute Prediction](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- **Size**: 821 images across 5 categories
- **Categories**: Blazer (159), Blouse (178), Dress (167), Skirt (153), Tee (164)
- **Split Ratio**: 70% training (521), 15% validation (150), 15% testing (150)
- **Note**: Original ResNet50 model trained on this dataset; current system uses CLIP for zero-shot classification

---

## Installation

### Prerequisites

- Python 3.12 or higher
- Virtual environment (recommended: venv or conda)
- Apple Silicon Mac (for MPS acceleration) or CUDA-compatible GPU

### Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AI-Wardrobe-Assistant.git
cd AI-Wardrobe-Assistant

# Create and activate virtual environment
python -m venv UF_AML
source UF_AML/bin/activate  # On macOS/Linux
# UF_AML\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.66.0
streamlit>=1.51.0
opencv-python
git+https://github.com/openai/CLIP.git
ftfy
regex

## Project Structure

AI-Wardrobe-Assistant/
├── ui/

├── src/
Start the Streamlit application:

```bash
# Option 1: Using startup script (recommended)
chmod +x start_ui.sh
./start_ui.sh

# Option 2: Direct launch
streamlit run ui/app_streamlit.py --server.port 8501
```

├── results/
**Access:** http://localhost:8501

**Note**: The UI file is now located in `ui/app_streamlit.py` (moved from root for better organization)

**Streamlit UI Features:**

**Classification & Analysis:**

- 📤 Image upload (drag-and-drop or file browser)
- 🎯 Real-time CLIP-based classification
- 📊 Top-3 predictions with confidence scores
  ├── data/
- 🔍 Ensemble breakdown (CLIP/Keyword/Path contribution)
- ✅ Smart validation status with corrections
- 👥 Automatic gender detection
  ├── datasets/
  ├── docs/

├── notebooks/
**Recommendation System:**

- 🔍 Visual similarity search across 6,844 items
  ├── requirements.txt
  ├── start_ui.sh
  ├── README.md
  ├── evaluate_deliverable3_v4.py
  ├── evaluate_system.py
  ├── generate_ieee_visualizations.py
- 💡 Explainable recommendations (style, color, material, occasion)
- 🎚️ Adjustable similarity threshold (0.0-1.0)
- 📈 Confidence levels (Excellent/Good/Fair/Weak)
- 🏷️ Category and gender filtering

**User Interface:**

- ⚡ Real-time processing feedback (~0.8s)
- 📊 Session statistics tracking
- 🎨 Clean, modern design with visual feedback
- 🔄 Auto-reload on code changes (development mode)
- 📱 Responsive layout for different screen sizes

---

## 📁 Project Structure

```
AI-Wardrobe-Assistant/
│
├── ui/                                # Web Interface
│   └── app_streamlit.py              # Main Streamlit application
│
├── src/                               # Core Modules
│   ├── ensemble_classifier.py        # CLIP + Keyword + Path fusion
│   ├── smart_validator.py            # Confidence-based validation
│   ├── recommender.py                # Cosine similarity search
│   ├── recommendation_explainer.py   # Explainable AI module
│   ├── model.py                      # ResNet50 classifier (legacy)
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── dataset_loader.py             # Data loading utilities
│   ├── build_gallery_index.py        # Gallery indexing tool
│   ├── download_polyvore.py          # Dataset download script
│   └── organize_deepfashion.py       # Dataset organization tool
│
├── results/                           # Model Outputs & Artifacts
│   ├── model_best.pth                # Trained ResNet50 (legacy)
│   ├── gallery_embeddings.npy        # CLIP embeddings
│   ├── gallery_index.npz             # Feature index
│   ├── gallery_meta.json             # Metadata (6,844 items)
│   ├── class_to_idx.json             # Category mapping
│   ├── evaluation_report.md          # Performance metrics
│   └── deliverable3_*.png            # Evaluation visualizations
│
├── data/                              # Datasets (~3.2GB)
│   ├── polyvore/                     # Polyvore outfit dataset
│   ├── dresscode/                    # Men's fashion dataset
│   └── deepfashion/                  # DeepFashion subset
│
├── datasets/                          # Kaggle datasets (~3.1GB)
│
├── docs/                              # Documentation
│   ├── DELIVERABLE3_COMPLETE_REPORT.md  # Comprehensive report
│   ├── STREAMLIT_GUIDE.md            # UI documentation
│   └── STREAMLIT_READY.md            # Deployment guide
│
├── notebooks/                         # Jupyter Notebooks
│
├── requirements.txt                   # Python dependencies
├── start_ui.sh                        # Quick launch script
├── README.md                          # 📖 This file
├── evaluate_deliverable3_v4.py        # Latest evaluation script
├── evaluate_system.py                 # System evaluation
└── generate_ieee_visualizations.py    # IEEE paper figures
```

**Key Components:**

- **CLIP System**: Zero-shot classification with no training required
- **Ensemble Classifier**: Weighted fusion of CLIP (95%), keyword (3%), path (2%)
- **Smart Validator**: Confidence-based filtering with error pattern detection
- **Gallery Index**: 6,000 items with pre-computed embeddings for fast similarity search

---

## Current System Architecture

### Classification Approach: CLIP + Ensemble + Smart Validator

**Phase 1: CLIP Zero-Shot Classification**

- Model: OpenAI CLIP ViT-B/32 (pre-trained on 400M image-text pairs)
- Zero-shot: No training required, direct visual-text matching
- Categories: 7 classes with enhanced descriptive prompts
- Feature extraction: 512-D embeddings for similarity search

**Phase 2: Ensemble Classifier**

- Multi-signal fusion: CLIP (95%) + Keyword (3%) + Path (2%)
- Smart strategy: Pure CLIP when no metadata, fusion only with strong signals
- Enhanced prompts: Detailed visual descriptions for better discrimination
- Example: "full length pants, jeans or trousers covering entire legs from waist down to ankles"

**Phase 3: Smart Validator**

- Confidence-based validation: High (>0.90), Medium (>0.70), Low (>0.50)
- Error pattern detection: Known confusion pairs (tee�blouse, pants�shorts)
- Consistency checks: Text-category alignment verification

### Performance Metrics (Current System)

| Component              | Baseline (Pure CLIP) | Ensemble + Validator  | Final System                  |
| ---------------------- | -------------------- | --------------------- | ----------------------------- |
| **Accuracy**     | 62.0%                | 68.0%                 | **73.47%** ✨           |
| **Test Set**     | 98 images            | 98 images             | 98 images                     |
| **Strategy**     | Simple prompts       | Multi-signal (95/3/2) | + Smart Validator + Explainer |
| **Thresholds**   | N/A                  | N/A                   | 0.90/0.70/0.50                |
| **Processing**   | ~0.5s                | ~0.7s                 | ~0.8s                         |
| **Gallery Size** | 6,000                | 6,000                 | **6,844**               |

### Optimization History

**Baseline (Pure CLIP):**

- Accuracy: 62.0%
- Simple category prompts (e.g., "a photo of a tee")
- No post-processing

**Phase 1+2 (Ensemble System):**

- Accuracy: 57.0%  (degraded due to weak signals)
- Multi-signal fusion: CLIP 85% + Keyword 10% + Path 5%
- Smart Validator: Thresholds 0.85/0.60/0.40
- Issue: Keyword/path noise reduced accuracy

**Current System (Deliverable 3):**

- **Accuracy**: 73.47% (+11.47% from baseline)
- **Architecture**: Ensemble (95/3/2) + Smart Validator + Explainer
- **Gallery**: 6,844 items with CLIP embeddings
- **Features**: Gender filtering, explainable recommendations, confidence-based validation

### Key Improvements

1. Ensemble Classifier: Multi-signal fusion (CLIP 95% + Keyword 3% + Path 2%)
2. Smart Validator: Category-specific correction rules, confidence thresholds (0.90/0.70/0.50)
3. Recommendation Explainer: Style, color, material, and occasion analysis
4. Gender Detection: Automatic filtering with category-specific rules
5. Enhanced Gallery: 6,844 items (+844 from baseline) with rich metadata

### Known Limitations

- Top/Bottom Confusion: Tee <-> Blouse, Pants <-> Shorts still occur at ~5% rate
- Processing Speed: ~0.8s per image (CLIP inference overhead)
- Gallery Coverage: Limited to 6,000 items (categories imbalanced - Tee 27%, Shorts 3.6%)
- No Training Data: Zero-shot approach, no domain-specific fine-tuning

---

## Technical Details

### System Architecture

**CLIP Visual Encoder (ViT-B/32)**:

- Pre-trained on 400M image-text pairs (OpenAI)
- Zero-shot classification via image-text similarity
- Output: 512-dimensional embeddings
- No training required - uses visual-semantic alignment

**Ensemble Classifier Pipeline**:

```python
# Multi-signal fusion
final_score = (
    0.95 * clip_score +        # Visual similarity (primary)
    0.03 * keyword_score +     # Text metadata (weak signal)
    0.02 * path_score          # Filename heuristic (minimal)
)

# Enhanced prompts for better discrimination
prompts = {
    'tee': "a photo of a short sleeve t-shirt, casual tee...",
    'pants': "full length pants, jeans or trousers covering..."
}
```

**Smart Validator (Confidence-Based)**:

- High confidence (>0.90): Accept immediately
- Medium confidence (0.70-0.90): Check known error patterns
- Low confidence (<0.70): Manual review recommended

### Recommendation Algorithm

1. **Feature Extraction**: Query image � CLIP encoder � 512-dim embedding
2. **Similarity Search**: Cosine similarity vs 6,000 gallery embeddings
3. **Category Filtering**: Same category as classified result
4. **Ranking**: Sort by similarity score (descending)
5. **Top-k Selection**: Return 5 most similar items

**Similarity Metric**:

```
similarity(query, gallery_item) = (query � gallery_item) / (||query|| × ||gallery_item||)
```

### Device Compatibility

The system automatically detects available hardware:

1. **MPS** (Metal Performance Shaders) - Apple Silicon Macs (recommended)
2. **CUDA** - NVIDIA GPUs
3. **CPU** - Fallback for universal compatibility

**Dependencies**:

- PyTorch 2.0+
- CLIP (OpenAI)
- Streamlit 1.28+
- Pillow, NumPy

---

## Development

### Future Enhancements

**Potential Optimization Strategies:**

- **Plan B** (5h effort): CLIP fine-tuning on fashion domain, hyperparameter grid search
- **Plan C** (10-13h effort): Advanced CLIP variants (ViT-L/14), attention-based fusion, ensemble learning

**Adding New Categories**:

1. Add images to `data/polyvore_items/{new_category}/`
2. Update category prompts in `src/ensemble_classifier.py`
3. Rebuild CLIP gallery index: `python build_clip_gallery.py`
4. Update metadata in `data/gallery_metadata.json`

Known Error Patterns:

- Tee <-> Blouse confusion: Similar silhouettes, need texture/collar cues
- Pants <-> Shorts confusion: Length discrimination at image boundaries

### Deployment Considerations

- **Batch Processing**: Enable batch CLIP inference for 10x speedup
- **Index Caching**: Pre-load `clip_gallery_index.pkl` to avoid cold start
- **API Wrapper**: Use FastAPI for RESTful service deployment
- **Model Hosting**: Hugging Face Spaces or Streamlit Cloud ready

---

## Deliverable 3 Highlights

### System Evolution

| Metric                     | Baseline (CLIP) | Deliverable 2    | Deliverable 3             | Improvement                    |
| -------------------------- | --------------- | ---------------- | ------------------------- | ------------------------------ |
| **Accuracy**         | 62.0%           | 56.6% (ResNet50) | **73.47%**          | **+16.87%**              |
| **Architecture**     | Pure CLIP       | ResNet50         | CLIP Ensemble + Validator | Multi-modal                    |
| **Gallery Size**     | 6,000           | 6,000            | **6,844**           | **+844 items**           |
| **Features**         | Basic           | Classification   | + Explainer + Validator   | Advanced                       |
| **Gender Detection** | No labels      | No labels        | Zero-shot                | +Gender intelligence     |

### New Capabilities

**1. Ensemble Classification:**

- Multi-signal fusion: CLIP (95%) + Keyword (3%) + Path (2%)
- Enhanced category prompts with detailed descriptions
- Smart fallback to pure CLIP when metadata is weak

**2. Smart Validation:**

- Confidence-based filtering (0.90/0.70/0.50 thresholds)
- Category-specific correction rules
- Known confusion pair detection (tee <-> blouse, pants <-> shorts)

**3. Explainable AI:**

- Style matching analysis (casual, formal, elegant, etc.)
- Color coordination detection
- Material and occasion inference
- Detailed reasoning for each recommendation

**4. Gender Intelligence:**

- Automatic gender detection from categories
- Category-specific rules (Blazer: Male, Blouse: Female, Tee: Unisex)
- Filtered recommendations based on detected gender

### Evaluation Results

Confusion Matrix: results/deliverable3_v4_confusion_matrix.png
Per-Class Metrics: results/deliverable3_v4_per_class_metrics.png
System Evolution: results/deliverable3_v4_evolution.png
Complete Report: docs/DELIVERABLE3_COMPLETE_REPORT.md

---

Known Issues:

1. Dataset Size

- Limited to 821 training images (prototype)
- Full DeepFashion dataset (13K+ images) not yet used
- May affect generalization to diverse clothing styles

2. Data Quality Issues

- Label Noise: DeepFashion labels contain errors
  - Multi-item images labeled with single category (e.g., skirt+sweater only "Skirt")
  - Mislabeled items (e.g., "Blazer" label but no blazer in image)
  - Ambiguous boundaries between similar categories (Blazer vs Blouse)
- Impact:
  - Model learns incorrect feature associations
  - Reduced classification accuracy
  - Lower embedding quality affects recommendation system
- Mitigation: Manual label verification recommended

3. Category Coverage

- Only 7 categories (Blazer, Blouse, Dress, Skirt, Tee, Pants, Shorts)
- Does not cover shoes, accessories, outerwear
- Western fashion bias

4. Background Sensitivity

- Works best with clean, centered product images
- Complex backgrounds may reduce accuracy
- Lighting variations can affect feature extraction

5. Gallery Index

- Requires 2-3 hours to build on Apple Silicon
- Must rebuild if model architecture changes

6. Training Stability

- Validation loss may fluctuate
- Overfitting risk on small dataset

Planned Fixes:

- Expand to full DeepFashion dataset
- Add background removal preprocessing
- Implement multi-label classification
- Optimize gallery indexing
- Add ONNX export for cross-platform deployment

---

Documentation:

- Complete Report: docs/DELIVERABLE3_COMPLETE_REPORT.md
- Training Notebooks: notebooks/deliverable3_ensemble_training.ipynb, notebooks/deliverable3_evaluation_v4.ipynb

---

Future Work:

- Multi-label classification (pattern, color, style attributes)
- User preference learning with feedback loops
- Outfit compatibility scoring (top-bottom pairing validation)
- Model compression for mobile deployment
- RESTful API using FastAPI
- Expand dataset and improve label quality
- Generative outfit synthesis using diffusion models
- Text-to-image retrieval with CLIP embeddings
- Temporal style trend analysis
- Sustainability metrics
- Cultural diversity expansion

---

References:

- DeepFashion: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- Polyvore Dataset: https://github.com/xthan/polyvore-dataset

---

License:
MIT License
Third-Party Licenses:

- PyTorch: BSD-3-Clause License
- DeepFashion Dataset: Academic use only
- Polyvore Dataset: Research purposes only

---

Acknowledgments:
Developed as part of advanced machine learning coursework.
Special thanks to:

- CUHK Multimedia Lab for DeepFashion dataset
- Polyvore team for outfit compatibility dataset
- PyTorch and Streamlit communities

---

Contact: tzuchiehchao@ufl.edu
Project Link: https://github.com/jayChao2431/AI-Wardrobe-Assistant
For questions or suggestions, please open an issue on GitHub.
