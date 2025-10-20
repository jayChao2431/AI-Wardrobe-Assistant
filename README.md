# AI-Powered-Wardrobe-Assistant

An **AI-driven virtual stylist** that helps users organize their wardrobe and generate daily outfit recommendations using the clothes they already own.

---

## Project Overview
Have you ever looked at your closet full of clothes and still felt like you had nothing to wear?  
This project builds an AI assistant that:
- Digitizes your wardrobe using image recognition
- Classifies each clothing item (e.g., shirt, jeans, shoes)
- Recommends stylish outfit combinations based on professional fashion datasets

By encouraging reuse of existing clothes, this app promotes **sustainable fashion** and **reduces decision fatigue**.

##  Datasets and Sources
- **DeepFashion** (classification & attributes): https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- (Optional) **Text2Human** (outfit visualization): https://github.com/yumingj/Text2Human


##  Repository Structure
```
data/           → Sample images (DeepFashion)
notebooks/      → Jupyter notebooks (setup, EDA)
src/            → Core data & model code
ui/             → Streamlit app
results/        → Outputs and figures
docs/            → Architecture diagram & UI wireframe
```

## Installation
```bash
git clone https://github.com/<your-username>/AI-Wardrobe-Assistant.git
cd AI-Wardrobe-Assistant
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

##  Quick Start
- Open the setup notebook to verify your environment:
```bash
jupyter notebook notebooks/setup.ipynb
```
- Run the demo UI (placeholder):
```bash
streamlit run ui/app.py
```

## Tech Stack
- Python 3.10+
- PyTorch / torchvision (image models)
- OpenCV / Pillow (image I/O)
- Streamlit / Gradio (user interface)
- Matplotlib (visualization)


## Author
* **Name:** Tzu-Chieh Chao

