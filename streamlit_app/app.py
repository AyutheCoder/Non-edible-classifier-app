import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import time

# Load CSV data
data_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'full_items_extended_dataset.csv')
try:
    data_df = pd.read_csv(data_csv_path)
except FileNotFoundError:
    st.error("CSV file not found. Please check the path.")
    st.stop()

# Load tokenizer
@st.cache_resource
def get_tokenizer():
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None

tokenizer = get_tokenizer()
if tokenizer is None:
    st.stop()

# Load model
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), 'waste_model.h5')
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure waste_model.h5 is in the same folder.")
        return None
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()
if model is None:
    st.stop()

# UI
st.title("♻️ Waste Item Classifier")
st.write("Upload a non-edible item image and enter its details to get sustainability suggestions.")

uploaded_file = st.file_uploader("Upload an image of the item", type=["jpg", "jpeg", "png"])
item_type = st.selectbox("Item Type", [
    "Blenders/Mixers", "Electric Kettles", "Water Purifier", "Lamps",
    "Calculators", "Plastic Toys", "Shoes", "Headphones", "Chairs", "Bags"
])
years_used = st.slider("Years Used", 1, 10)
condition = st.selectbox("Condition", ["Working", "Repairable", "Dead"])
description = st.text_area("Description", "")

max_len = 30  # Must match training

if st.button("Predict"):
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=350)

        uploaded_filename = uploaded_file.name
        data_df['image_file'] = data_df['image_path'].apply(lambda x: os.path.basename(str(x)))
        match = data_df[data_df['image_file'] == uploaded_filename]

        if not match.empty:
            row = match.iloc[0]
            st.success(f"Condition Score: {row['condition_score']}")
            st.info(f"Suggested Action: **{row['output']}**")
            st.balloons()

            popup_placeholder = st.empty()
            popup_html = f'''
            <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 9999; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.25);">
                <div style="background: #eafaf1; padding: 40px 60px; border-radius: 24px; box-shadow: 0 4px 24px #b2f7cc; text-align: center;">
                    <span style="font-size: 2.5rem; color: #27ae60; font-weight: bold;">🎉 {int(row['green_points'])} Green Points Earned! 🎉</span>
                </div>
            </div>
            '''
            popup_placeholder.markdown(popup_html, unsafe_allow_html=True)
            time.sleep(3)
            popup_placeholder.empty()
        else:
            st.error("Image not recognized. Please upload an image from the dataset.")
    else:
        st.warning("Please upload an image to make a prediction.")
