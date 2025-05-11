import streamlit as st
import json
from PIL import Image
import numpy as np
import os
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Placeholder for model import - replace with your actual model loading
# from your_model_file import load_model, predict_food

# Debug: Print current working directory and file path
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Script directory: {os.path.dirname(__file__)}")

def load_model(config_path, model_path):
    # Debug: Print the paths being used
    st.write(f"Looking for config at: {config_path}")
    st.write(f"Looking for model at: {model_path}")

    # Check if config file exists
    if not os.path.exists(config_path):
        st.error(f"Config file not found at: {config_path}")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None

    try:
        # Load the model and processor
        model = ViTForImageClassification.from_pretrained(os.path.dirname(model_path))
        processor = ViTImageProcessor.from_pretrained(os.path.dirname(model_path))
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_food(model, processor, image):
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class].item()
        
        # Get the label from the model's config
        predicted_food = model.config.id2label[predicted_class]
        
        # Consider it a food item if confidence is above threshold
        is_food_item = confidence > 0.5
        
        return predicted_food, confidence, is_food_item
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Streamlit app
st.title("Indian Food Classifier")
st.write("Upload an image of Indian food to predict its name.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load the model using relative paths to the subdirectory
    config_path = os.path.join(os.path.dirname(__file__), "indian_food_finetuned_model-20250505T114517Z-001", "indian_food_finetuned_model", "config.json")
    model_path = os.path.join(os.path.dirname(__file__), "indian_food_finetuned_model-20250505T114517Z-001", "indian_food_finetuned_model", "model.safetensors")
    model, processor = load_model(config_path, model_path)

    # Predict the food name
    if model is not None and processor is not None and st.button("Predict"):
        food_name, confidence, is_food_item = predict_food(model, processor, image)
        
        if food_name is None:
            st.error("Failed to make prediction. Please try again.")
        elif not is_food_item:
            st.warning("This does not appear to be a food item.")
        elif confidence < 0.25:
            st.warning(f"Confidence is too low ({confidence:.2f}). Unable to predict the food item.")
        else:
            st.success(f"Predicted Food: **{food_name}** (Confidence: {confidence:.2f})")

# Instructions for running the app
st.markdown("### How to Run")
st.write("1. Ensure your model files are in `indian_food_finetuned_model-20250505T114517Z-001\\indian_food_finetuned_model\\` relative to this script.")
st.write("2. Install requirements: `pip install streamlit pillow numpy`")
st.write("3. Run the app from the correct directory: `cd C:\\Users\\sudik\\OneDrive\\Desktop\\rajesh && streamlit run app.py`")