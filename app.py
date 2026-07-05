import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",  
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.prediction-box{
    padding:20px;
    border-radius:12px;
    background-color:#f4f4f4;
    border:1px solid #dddddd;
}

.footer{
    font-size:13px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/bone_model.keras")

model = load_model()

IMG_SIZE = (224,224)

# -------------------------------
# TITLE
# -------------------------------
st.title("🦴 Bone Fracture Detection")
st.write(
    "Upload an X-ray image and let the AI model predict whether a fracture is present."
)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:

    st.header("📌 About")

    st.write("""
This application uses a deep learning model based on **EfficientNet**
to classify bone X-ray images.

**Model**
- EfficientNet
- TensorFlow/Keras

**Input Size**
- 224 × 224

**Output**
- Fracture
- Normal
""")

    st.markdown("---")

    st.info(
        "⚠ This tool is for educational purposes only and should not replace professional medical diagnosis."
    )

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("Prediction")

        with st.spinner("Analyzing X-ray..."):

            img = image.resize(IMG_SIZE)

            img = np.array(img)

            img = preprocess_input(img)

            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)

            probability = float(prediction[0][0])

            if probability >= 0.5:

                label = "🦴 Fracture Detected"

                confidence = probability

                color = "red"

            else:

                label = "✅ Normal Bone"

                confidence = 1 - probability

                color = "green"

        st.markdown(
            f"""
            <div class="prediction-box">
            <h3 style="color:{color};">{label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Confidence")

        st.progress(int(confidence*100))

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence*100:.2f}%"
        )

st.markdown("---")

st.warning(
"""
### Medical Disclaimer

This AI model is intended **only for educational and research purposes**.

It should **not** be used as a substitute for professional medical advice,
diagnosis, or treatment.
"""
)

st.markdown(
"""
<div class='footer'>

Developed using ❤️ with TensorFlow & Streamlit

</div>
""",
unsafe_allow_html=True
)