# terrain_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
from io import BytesIO
# use_column_width
# Set page config
st.set_page_config(
    page_title="Terrain Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and class indices
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("terrain_classifier.h5")  # Updated to .h5
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
    return model, class_names

model, class_names = load_model_and_classes()

# Preprocess image
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict_terrain(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)[0]
    return predictions

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        color: #ffffff;
        font-size: 3.5em;
        font-weight: bold;
    }
    .subtitle {
        color: #34495e;
        font-size: 2.2em;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    st.subheader("About")
    st.write("""
    This app classifies terrain types using a deep learning model trained on the EuroSAT dataset.
    Upload an image to get started!
    """)

# Main content
st.markdown('<h1 class="title">Terrain Classification App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to classify terrain type</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a satellite image for classification"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    predictions = predict_terrain(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Display prediction
    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        
        if confidence >= confidence_threshold:
            st.success(f"Predicted Terrain: {predicted_class}")
            st.write(f"Confidence: {confidence*100:.2f}%")
        else:
            st.warning("Low confidence prediction")
            st.write(f"Most likely: {predicted_class} ({confidence*100:.2f}%)")
        
        # Confidence meter
        st.progress(float(confidence))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed prediction chart
    st.subheader("Prediction Distribution")
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions,
            marker_color=['#2ecc71' if i == predicted_class_idx else '#3498db' 
                         for i in range(len(class_names))]
        )
    ])
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Terrain Types",
        yaxis_title="Probability",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download prediction
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    st.download_button(
        label="Download Prediction",
        data=buffer.getvalue(),
        file_name=f"prediction_{predicted_class}.png",
        mime="image/png"
    )

# Footer
st.markdown("---")
st.write("")
