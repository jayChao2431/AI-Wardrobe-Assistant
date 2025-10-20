# 👗 AI-Wardrobe-Assistant

An **AI-driven virtual stylist** that helps users organize their wardrobe, classify clothes, and generate stylish outfit recommendations — extensible to visualize outfits in the future.

## 🧠 Project Overview
Users can upload photos of their clothes, and the system will:
1) **Classify** the item (category, color, style)
2) **Recommend** compatible pieces based on real-world fashion data
3) (Future) **Visualize** outfits on a human figure (e.g., Text2Human)

This project promotes **sustainable fashion** and reduces decision fatigue.

## 📊 Datasets and Sources
- **DeepFashion** (classification & attributes): https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- (Optional) **Text2Human** (outfit visualization): https://github.com/yumingj/Text2Human

> This template includes DeepFashion *sample images* only for Deliverable 1 verification.

## 🧩 Repository Structure
```
data/           → Sample images (DeepFashion)
notebooks/      → Jupyter notebooks (setup, EDA)
src/            → Core data & model code
ui/             → Streamlit app
results/        → Outputs and figures
docs/            → Architecture diagram & UI wireframe
```

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/AI-Wardrobe-Assistant.git
cd AI-Wardrobe-Assistant
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 🚀 Quick Start
- Open the setup notebook to verify your environment:
```bash
jupyter notebook notebooks/setup.ipynb
```
- Run the demo UI (placeholder):
```bash
streamlit run ui/app.py
```

## 🧰 Tech Stack
- Python 3.10+
- PyTorch / torchvision (image models)
- OpenCV / Pillow (image I/O)
- Streamlit / Gradio (user interface)
- Matplotlib (visualization)

## 🌱 Responsible AI
All processing can be done locally. No personal data is uploaded. This project encourages reuse of existing clothes to reduce waste.

## 🧑‍💻 Author
**Jay Chao**  
Deliverable 1 – From Pitch to Prototype
