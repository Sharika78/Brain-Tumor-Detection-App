import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Config
st.set_page_config(
    page_title="Brain Tumor MRI Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification System")
st.write("Upload a Brain MRI Scan to analyze and detect tumor types.")

# Load Trained Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

model = load_model()

# Class labels
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# File Uploader
uploaded_file = st.file_uploader("Choose a Brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    st.write("🔍 **Analyzing MRI Image...**")
    
    # Preprocessing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_reshape)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    
    st.divider()
    
    # Show Results
    if predicted_class == 'No Tumor':
        st.success(f"✅ **Result:** {predicted_class}")
    else:
        st.error(f"🚨 **Result Detected:** {predicted_class}")
        
    st.info(f"📊 **Model Confidence:** {confidence:.2f}%")
# Tumor Information Dictionary
    tumor_info = {
        'Glioma Tumor': 'Gliomas grow in the glial cells of the brain. They are one of the most common types of primary brain tumors.',
        'Meningioma Tumor': 'Meningiomas start in the membranes that protect the brain and spinal cord. Most are slow-growing and benign.',
        'Pituitary Tumor': 'Pituitary tumors form in the pituitary gland at the base of the brain and can affect hormone balance.',
        'No Tumor': 'No brain tumor detected in this MRI scan. The scan appears clear.'
    }

    # Display Description
    if predicted_class in tumor_info:
        st.subheader("📌 Description & Details")
        st.info(tumor_info[predicted_class])