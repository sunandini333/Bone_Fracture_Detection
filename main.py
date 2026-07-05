from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

# Create FastAPI app
app = FastAPI(
    title="Bone Fracture Detection API",
    description="API for Bone Fracture Prediction using EfficientNet",
    version="1.0"
)

# Load the trained model
model = load_model("BoneFracture_EfficientNet.keras")

# Image size used during training
IMG_SIZE = (224, 224)

# Class names (change these if your folder names are different)
CLASS_NAMES = ["Normal", "Fracture"]


@app.get("/")
def home():
    return {
        "message": "Bone Fracture Detection API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded X-ray contains a bone fracture.
    """

    # Read uploaded image
    image = Image.open(file.file).convert("RGB")

    # Resize image
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array
    image = np.array(image)

    # Preprocess for EfficientNet
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        predicted_class = CLASS_NAMES[1]
    else:
        predicted_class = CLASS_NAMES[0]

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "probability": probability
    }