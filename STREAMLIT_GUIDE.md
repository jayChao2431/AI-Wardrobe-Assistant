# Streamlit UI Installation and Usage Guide

## Step 1: Streamlit Installation

Execute the following command in your terminal:

```bash
cd ~/Documents/GitHub/AI-Wardrobe-Assistant
/Users/chaotzuchieh/Documents/GitHub/UF_AML/bin/pip install streamlit
```

Expected output:
```
Collecting streamlit
  Downloading streamlit-1.x.x-py2.py3-none-any.whl
...
Successfully installed streamlit-1.x.x
```

---

## Step 2: Application Launch

### Method 1: Using Absolute Path
```bash
cd ~/Documents/GitHub/AI-Wardrobe-Assistant
/Users/chaotzuchieh/Documents/GitHub/UF_AML/bin/streamlit run app_streamlit.py
```

### Method 2: Activate Virtual Environment First
```bash
cd ~/Documents/GitHub/AI-Wardrobe-Assistant
source ~/Documents/GitHub/UF_AML/bin/activate
streamlit run app_streamlit.py
```

---

## Step 3: Browser Access

After launching the application, the terminal will display:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Automatic Launch:** Streamlit typically opens the browser automatically  
**Manual Access:** Navigate to `http://localhost:8501` in your web browser

---

## Application Features

### Core Functionalities:
1. **Image Upload**
   - Supports JPG, JPEG, PNG formats
   - Option to select example images from test dataset

2. **Real-time Analysis**
   - Clothing category recognition (Blazer, Blouse, Dress, Skirt, Tee)
   - Confidence score visualization
   - Top-3 prediction results display

3. **Intelligent Recommendations**
   - Similarity-based recommendation of 3-10 matching items
   - Visual presentation of recommendation results
   - Similarity scoring with quality indicators

4. **Configurable Settings**
   - Adjustable recommendation count (3-10 items)
   - Category filtering options
   - Model information display

---

## Important Prerequisites

### Gallery Index Construction Required

If `build_gallery_index.py` has not been executed, the application will display an error:

```
⚠️ Gallery index not found! 
Please run: python src/build_gallery_index.py
```

**Resolution:**
```bash
cd ~/Documents/GitHub/AI-Wardrobe-Assistant
python src/build_gallery_index.py
```

This procedure will:
- Extract features from 252K Polyvore images
- Generate `results/gallery_index.npz` and `results/gallery_meta.json`
- Require approximately 2-3 hours (utilizing MPS GPU acceleration)

---

## User Interface Description

### Left Panel (Sidebar)
- **Settings:** Adjust recommendation count and filtering options
- **Model Information:** Display model architecture and database statistics
- **About:** System description and dataset information

### Main Panel - Left Section
- **Upload Image:** Upload custom images or select example images
- **Analyze Button:** Execute classification and recommendation analysis
- **Classification Results:** Display prediction outcomes with confidence scores

### Main Panel - Right Section
- **Recommendations:** Present recommendation results
- Each recommendation displays:
  - Image thumbnail
  - Similarity score
  - Match quality rating (Excellent/Good/Fair/Weak)

---

## Usage Examples

### Example 1: Upload Custom Image
1. Click "Choose a clothing image..."
2. Select your clothing photograph
3. Click "Analyze & Recommend"
4. Review classification results and recommended items

### Example 2: Use Test Dataset Images
1. Select a category from the dropdown menu (e.g., Blazer)
2. The system automatically loads an example image from that category
3. Click "Analyze & Recommend"
4. Review recommendation results

---

## Advanced Configuration

### Modifying Default Parameters

Edit the `app_streamlit.py` file:

```python
# Adjust default recommendation count
num_recommendations = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5,  # ← Modify this value
    step=1
)

# Customize page title
st.set_page_config(
    page_title="AI Wardrobe Recommender",  # ← Modify this text
    page_icon="👔",
    layout="wide"
)
```

### Customizing UI Color Scheme

Modify the Custom CSS block:

```python
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;  # ← Change header color
    }
</style>
""", unsafe_allow_html=True)
```

---

## Troubleshooting

### Q1: Application Launch Failure
**Error:** `streamlit: command not found`

**Solution:**
```bash
# Use absolute path
/Users/chaotzuchieh/Documents/GitHub/UF_AML/bin/streamlit run app_streamlit.py
```

### Q2: Model Loading Error
**Error:** `FileNotFoundError: model_best.pth`

**Solution:** Verify model training completion
```bash
# Check file existence
ls results/model_best.pth
ls results/class_to_idx.json
```

### Q3: Gallery Index Not Found
**Error:** `Gallery index not found!`

**Solution:** Build feature database
```bash
python src/build_gallery_index.py
```

### Q4: Out of Memory Error
**Error:** `Out of Memory`

**Solution:** Reduce `BATCH_SIZE` in `build_gallery_index.py`
```python
BATCH_SIZE = 16  # Decrease batch size
```

### Q5: Slow Recommendation Loading
**Cause:** Large gallery database (252K images)

**Solution:** This is expected behavior. Initial loading requires time; subsequent queries utilize caching.

---

## Conclusion

You now have a complete interactive AI-powered wardrobe recommendation system.

### Key Features:
- Modern web interface  
- Real-time image analysis  
- Intelligent recommendation algorithm  
- Interactive configuration options  
- Visual recommendation display  

### Recommended Screenshots:
1. Main interface overview
2. Image upload with classification results
3. Recommendation results display
4. Sidebar configuration panel

---

## Related Files

- `app_streamlit.py` - Streamlit UI main application
- `demo_recommendation.py` - Command-line recommendation system
- `src/build_gallery_index.py` - Feature database construction tool
- `src/recommender.py` - Core recommendation algorithm

---

## Application Termination

Press `Ctrl + C` in the terminal to stop the Streamlit server.

---

## Next Steps

1. ✅ Install Streamlit
2. ✅ Launch application
3. ✅ Test functionality
4. 📸 Capture demonstration screenshots
5. 📝 Prepare project documentation
