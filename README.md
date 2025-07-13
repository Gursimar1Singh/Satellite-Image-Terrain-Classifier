# Terrain Classification Web App 

An interactive Streamlit-based web application that classifies satellite images into terrain types using a deep learning model trained on the EuroSAT dataset.

## Features

- **Image Upload & Classification:** Upload satellite images (JPEG/PNG) and get terrain classification predictions in real time.
- **Confidence Thresholding:** Customize confidence sensitivity using a slider.
- **Prediction Visualization:**
  - Displays predicted terrain class and confidence score.
  - Plots class-wise prediction probabilities using Plotly.
- **Download Option:** Download the input image with the prediction label.
- **Responsive UI:** Clean two-column layout with modern CSS for a user-friendly interface.

## How It Works

1. Upload a satellite image via the interface.
2. The image is preprocessed and passed through a trained Keras model (`terrain_classifier.h5`).
3. The model outputs class probabilities.
4. The most confident class is selected and visualized with:
   - A success/warning message
   - A prediction bar chart
   - A confidence meter

## Project Structure
```
terrain-classifier/
├── terrain_app.py # Main Streamlit app
├── terrain_classifier.h5 # Trained Keras model
├── class_indices.json # Mapping of class names to model outputs
├── requirements.txt # Python dependencies
```

## Tech Stack

- **Frontend/UI:** Streamlit, HTML/CSS
- **Backend/ML:** TensorFlow, Keras, NumPy, Pillow
- **Visualization:** Plotly
- **Model:** CNN trained on the EuroSAT dataset (10 terrain classes)

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/terrain-classifier.git
    cd terrain-classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run terrain_app.py
    ```

## Sample Output

Once the image is uploaded:
- The terrain type is predicted (e.g., Forest, River, Residential).
- A probability distribution chart is displayed.
- Download option appears for the prediction result.

## Future Enhancements

- Add support for batch image classification
- Show prediction overlays on satellite images
- Integrate GPS/map-based image fetching
- Add terrain-specific recommendations (e.g., for agriculture, urban planning)

