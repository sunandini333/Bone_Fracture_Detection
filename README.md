# 🦴 Bone Fracture Detection

An AI-powered deep learning application that automatically detects bone fractures from X-ray images using **EfficientNetB0** and provides fast predictions through a **FastAPI** backend.

## 📌 Project Overview

Bone fracture diagnosis from X-ray images is a time-consuming process that requires expert radiologists. This project leverages **Convolutional Neural Networks (CNNs)** with **Transfer Learning** to classify X-ray images as **Fractured** or **Not Fractured**, helping support faster and more consistent preliminary diagnosis.

---

## 🚀 Features

* Bone fracture detection from X-ray images
* EfficientNetB0 transfer learning model
* Image preprocessing and augmentation
* FastAPI REST API for inference
* Easy-to-use prediction endpoint
* TensorFlow/Keras implementation
* Ready for deployment

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* EfficientNetB0
* FastAPI
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pillow
* Uvicorn

---

## 📂 Project Structure

```text
Bone_Fracture_Detection/
│
├── data/
│   ├── train/
│   └── val/
│
├── models/
│   └── bone_model.keras
│
├── notebooks/
│   └── CNN_Model.ipynb
|   └── Transfer_learning.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset contains X-ray images categorized into two classes:

* Fractured
* Not Fractured

The images are resized to **224 × 224 pixels** before being passed to the model.

---

## 🧠 Model

This project uses **EfficientNetB0** as the backbone network with transfer learning.

### Training Pipeline

* Image resizing (224 × 224)
* Data augmentation
* EfficientNetB0 pretrained on ImageNet
* Global Average Pooling
* Dense classification layer
* Binary classification

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/sunandini333/Bone_Fracture_Detection.git

cd Bone_Fracture_Detection
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run the training notebook or training script to generate the trained model.

---

## 🚀 Run the Streamlit App

Start the application using:

```bash
streamlit run app.py

---

## 🌐 Live Demo

Try the application here:

**(https://bonefracturedetection-ndowdhazdf9gofeqzmkeyp.streamlit.app/)**

---

## 📈 Results

You can find the website at [Website](https://bonefracturedetection-ndowdhazdf9gofeqzmkeyp.streamlit.app/)

---

## 📸 Sample Predictions


*<img width="720" height="434" alt="image" src="https://github.com/user-attachments/assets/6b5bc334-55df-47ff-ad89-3ec822e982f3" />
*

---

## 🔮 Future Improvements

* Multi-class fracture detection
* Fracture localization using object detection
* Explainable AI using Grad-CAM
* Cloud deployment
* Docker support
* Web interface

---

## 👩‍💻 Author

**Sunandini Chowdhury**

M.Tech Computational Biology
IIIT Delhi

GitHub: https://github.com/sunandini333

---

## ⭐ If you found this project useful

Please consider giving the repository a **Star ⭐**.




