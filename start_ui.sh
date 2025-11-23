#!/bin/bash

# AI-Powered Wardrobe Recommender - Quick Start Script
# This script launches the Streamlit web application

echo "🚀 Starting AI-Powered Wardrobe Recommender..."
echo ""

# Change to project directory
cd ~/Documents/GitHub/AI-Wardrobe-Assistant

# Check if gallery index exists
if [ ! -f "results/gallery_index.npz" ]; then
    echo "⚠️  WARNING: Gallery index not found!"
    echo ""
    echo "The recommendation system requires a feature database."
    echo "Please run the following command first:"
    echo ""
    echo "  python src/build_gallery_index.py"
    echo ""
    echo "This will take 2-3 hours to process 252K images."
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "results/model_best.pth" ]; then
    echo "❌ ERROR: Trained model not found!"
    echo ""
    echo "Please complete model training first:"
    echo "  Run the training notebook: notebooks/train_and_evaluate_detailed.ipynb"
    echo ""
    exit 1
fi

# Launch Streamlit
echo "✅ All checks passed!"
echo ""
echo "🌐 Launching Streamlit web application..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

/Users/chaotzuchieh/Documents/GitHub/UF_AML/bin/streamlit run ui/app_streamlit.py
