# AI-Powered Wardrobe Recommender

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent clothing recognition and recommendation system that leverages deep learning for garment classification and feature-based similarity matching to provide personalized outfit suggestions.

---

## Abstract

This project implements a complete AI-driven wardrobe management system that addresses the common challenge of outfit selection through automated clothing categorization and intelligent recommendation algorithms. The system employs transfer learning with ResNet50 architecture trained on the DeepFashion dataset for feature extraction, combined with cosine similarity-based retrieval on the Polyvore outfit dataset for style-aware recommendations. The implementation includes both command-line and web-based interfaces, enabling practical deployment for real-world fashion applications.

---

## Features

- **Deep Learning Classification**: ResNet50-based transfer learning model trained for 5-category clothing recognition (Blazer, Blouse, Dress, Skirt, Tee)
- **Feature-Based Recommendation**: 2048-dimensional feature embeddings with cosine similarity matching across 252K fashion items
- **Interactive Web Interface**: Streamlit-powered UI with real-time classification and dynamic recommendation visualization
- **Comprehensive Pipeline**: End-to-end workflow from data preprocessing to model deployment and evaluation
- **Apple Silicon Optimization**: Native MPS (Metal Performance Shaders) GPU acceleration for efficient inference
- **Extensible Architecture**: Modular design supporting multiple datasets, model architectures, and deployment strategies

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input Layer                        │
│  (Image Upload / File Browser / Camera Capture)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Preprocessing Pipeline                        │
│  • Resize to 224×224                                        │
│  • Normalization (ImageNet statistics)                     │
│  • Tensor conversion                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Feature Extraction Module                        │
│  ResNet50 (ImageNet pre-trained)                           │
│  • Frozen backbone (layers 1-3)                            │
│  • Trainable block (layer 4)                               │
│  • Fully connected classifier                              │
│  Output: (logits, 2048-dim features)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ├─────────────────┬───────────────────────┐
                      ▼                 ▼                       ▼
          ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
          │ Classification │  │ Feature Database │  │  Recommendation  │
          │    Engine      │  │   (Polyvore)     │  │     Engine       │
          │  Softmax + Top-K│  │  252K embeddings │  │ Cosine Similarity│
          └────────┬───────┘  └────────┬─────────┘  └────────┬─────────┘
                   │                   │                      │
                   └─────────────────┬─┴──────────────────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │   Presentation Layer     │
                      │ • Streamlit Web UI       │
                      │ • CLI Demo Tool          │
                      │ • Visualization Export   │
                      └──────────────────────────┘
```

---

## Dataset Specifications

### DeepFashion Subset (Training & Evaluation)
- **Source**: [DeepFashion: Category and Attribute Prediction](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- **Size**: 821 images across 5 categories
- **Categories**: Blazer (159), Blouse (178), Dress (167), Skirt (153), Tee (164)
- **Split Ratio**: 70% training (521), 15% validation (150), 15% testing (150)
- **Resolution**: Variable (resized to 224×224 during preprocessing)
- **Purpose**: Multi-class classification and supervised feature learning

### Polyvore Outfits (Recommendation Gallery)
- **Source**: [Polyvore Dataset](https://github.com/xthan/polyvore-dataset)
- **Size**: 252,068 fashion images
- **Content**: Real-world outfit combinations curated by fashion experts
- **Purpose**: Feature database for similarity-based recommendation
- **Index Format**: NumPy compressed array (252068 × 2048 float32)

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
```

---

## Usage

### 1. Model Training

Train the ResNet50 classification model on DeepFashion subset:

```bash
# Using Jupyter Notebook (recommended for exploration)
jupyter notebook notebooks/train_and_evaluate_detailed.ipynb

# Or run training script directly
python src/train_model.py \
    --data_dir data/deepfashion_subset \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device mps
```

**Training Output**:
- Model checkpoint: `results/model_best.pth` (90 MB)
- Training curves: `results/training_curves.png`
- Test metrics: `results/test_metrics.json`
- Confusion matrices: `results/confusion_matrix*.png`

### 2. Gallery Index Construction

Build feature database from Polyvore dataset (2-3 hours on MPS GPU):

```bash
python src/build_gallery_index.py
```

**Processing Details**:
- Batch size: 32 images
- Feature dimension: 2048
- Expected duration: 2-3 hours for 252K images
- Output files:
  - `results/gallery_index.npz` (~2 GB)
  - `results/gallery_meta.json`

### 3. Command-Line Demonstration

Test recommendation system with single image:

```bash
python demo_recommendation.py path/to/test/image.jpg
```

**Output**: `results/recommendation_demo.png` showing query image and top-5 recommendations

### 4. Web Interface Launch

Start interactive Streamlit application:

```bash
# Option 1: Direct launch
streamlit run app_streamlit.py

# Option 2: Using startup script with validation
chmod +x start_ui.sh
./start_ui.sh

# Option 3: Specify port
streamlit run app_streamlit.py --server.port 8501
```

**Web UI Features**:
- File upload with drag-and-drop support
- Example image selection from test dataset
- Real-time classification with confidence scores
- Top-3 prediction visualization
- Adjustable recommendation count (3-10 items)
- Match quality indicators (Excellent/Good/Fair/Weak)

Access at: http://localhost:8501

---

## Project Structure

```
AI-Wardrobe-Assistant/
│
├── data/                           # Datasets
│   ├── deepfashion_subset/         # Training data (821 images)
│   │   ├── train/                  # 521 training images
│   │   ├── val/                    # 150 validation images
│   │   └── test/                   # 150 test images
│   └── polyvore/                   # Gallery data (252K images)
│
├── src/                            # Core modules
│   ├── model.py                    # ResNet50Classifier definition
│   ├── dataset_loader.py           # PyTorch data loading utilities
│   ├── recommender.py              # Cosine similarity ranking
│   ├── build_gallery_index.py      # Feature extraction pipeline
│   ├── download_polyvore.py        # Kaggle dataset downloader
│   └── organize_deepfashion.py     # Dataset organization tool
│
├── notebooks/                      # Jupyter notebooks
│   └── train_and_evaluate_detailed.ipynb  # Complete training pipeline
│
├── results/                        # Model outputs
│   ├── model_best.pth              # Trained model weights (90 MB)
│   ├── class_to_idx.json           # Category mappings
│   ├── training_curves.png         # Loss/accuracy plots
│   ├── confusion_matrix.png        # Classification evaluation
│   ├── test_metrics.json           # Performance statistics
│   ├── gallery_index.npz           # Feature database (2 GB)
│   └── gallery_meta.json           # Image metadata
│
├── app_streamlit.py                # Streamlit web application
├── demo_recommendation.py          # CLI demonstration tool
├── start_ui.sh                     # Quick launch script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── STREAMLIT_GUIDE.md              # UI usage documentation
├── STREAMLIT_READY.md              # Quick start reference
└── NEXT_STEPS.md                   # Development roadmap
```

---

## Model Performance

### Classification Metrics (DeepFashion Subset Test Set)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Blazer   | 0.92      | 0.89   | 0.91     | 30      |
| Blouse   | 0.87      | 0.91   | 0.89     | 30      |
| Dress    | 0.94      | 0.93   | 0.94     | 30      |
| Skirt    | 0.89      | 0.88   | 0.89     | 30      |
| Tee      | 0.91      | 0.93   | 0.92     | 30      |
| **Avg**  | **0.91**  | **0.91**| **0.91**| **150** |

### Training Configuration

| Parameter         | Value                  |
|-------------------|------------------------|
| Architecture      | ResNet50 (ImageNet)    |
| Optimizer         | Adam                   |
| Learning Rate     | 1e-3                   |
| Batch Size        | 32                     |
| Epochs            | 10                     |
| Device            | MPS (Apple Silicon)    |
| Training Time     | ~15 minutes            |

### Data Augmentation

- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
- Random affine transformation (scale=0.9-1.1, translate=10%)

---

## Technical Details

### Model Architecture

**ResNet50Classifier** extends PyTorch's pre-trained ResNet50:
- **Backbone**: ResNet50 layers 1-4 (conv1 → layer4)
- **Transfer Learning Strategy**: 
  - Frozen: conv1, layer1, layer2, layer3
  - Trainable: layer4 (fine-tuning)
- **Classifier Head**: 
  - Input: 2048-dimensional features
  - Output: 5-class logits + feature embeddings
- **Forward Pass**: Returns (logits, features) tuple

### Recommendation Algorithm

1. **Feature Extraction**: Query image → ResNet50 → 2048-dim embedding
2. **Similarity Computation**: Cosine similarity between query and gallery features
3. **Ranking**: Sort by similarity score (descending)
4. **Filtering**: Return top-k recommendations (default k=5)
5. **Visualization**: Display with match quality indicators

**Similarity Metric**:
```
similarity(query, gallery_item) = (query · gallery_item) / (||query|| × ||gallery_item||)
```

### Device Compatibility

The system automatically detects available hardware:
1. **MPS** (Metal Performance Shaders) - Apple Silicon Macs
2. **CUDA** - NVIDIA GPUs
3. **CPU** - Fallback for universal compatibility

---

## Development

### Adding New Categories

1. Organize images into `data/deepfashion_subset/train/{category_name}/`
2. Update category mappings if needed
3. Retrain model with adjusted number of output classes
4. Rebuild gallery index with updated model

### Custom Model Architectures

Modify `src/model.py` to experiment with:
- EfficientNet variants
- Vision Transformers (ViT)
- Custom CNN architectures

Ensure forward pass returns `(logits, features)` tuple for compatibility.

### Deployment Considerations

- **Model Compression**: Apply quantization to reduce model size
- **Batch Processing**: Enable batch inference for multiple images
- **Caching**: Implement Redis for frequently accessed recommendations
- **API Wrapper**: Use FastAPI for RESTful service deployment

---

## Future Work

### Short-term Enhancements
- [ ] Multi-label classification (pattern, color, style attributes)
- [ ] User preference learning with feedback loops
- [ ] Outfit compatibility scoring (top-bottom pairing validation)
- [ ] Mobile application deployment (iOS/Android)

### Long-term Research Directions
- [ ] Generative outfit synthesis using StyleGAN/Diffusion models
- [ ] Text-to-image retrieval with CLIP embeddings
- [ ] Temporal style trend analysis
- [ ] Sustainability metrics (CO₂ footprint, circular fashion scoring)

---

## References

### Academic Papers

1. Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). **DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations**. *Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

2. Han, X., Wu, Z., Jiang, Y. G., & Davis, L. S. (2017). **Learning Fashion Compatibility with Bidirectional LSTMs**. *Proceedings of ACM International Conference on Multimedia*.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. *Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

### Datasets

- **DeepFashion**: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- **Polyvore Dataset**: https://github.com/xthan/polyvore-dataset

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

### Third-Party Licenses
- PyTorch: BSD-3-Clause License
- DeepFashion Dataset: Academic use only (see dataset website)
- Polyvore Dataset: Research purposes (see dataset repository)

---

## Acknowledgments

This project was developed as part of advanced machine learning coursework. Special thanks to:
- CUHK Multimedia Lab for the DeepFashion dataset
- Polyvore team for the outfit compatibility dataset
- PyTorch community for deep learning framework
- Streamlit team for the web interface toolkit

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the project maintainers.

**Project Link**: https://github.com/yourusername/AI-Wardrobe-Assistant
