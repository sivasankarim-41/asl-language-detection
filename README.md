# 🤟 ASL Language Detection using CNN

## 📌 Overview

This project implements a real-time American Sign Language (ASL) gesture recognition system using Computer Vision and Deep Learning.
It captures hand gestures through a webcam, processes the image using OpenCV, and predicts the corresponding ASL alphabet using a trained Convolutional Neural Network (CNN).

---

## 🚀 Features

* Real-time hand detection using **cvzone**
* Image preprocessing and cropping using **OpenCV**
* Deep learning model for gesture classification
* Supports **29 classes** (A–Z + special symbols like *space, delete, nothing*)
* Live prediction displayed on webcam feed

---

## 🛠️ Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* cvzone

---

## 📂 Project Structure

```
asl-language-detection/
│
├── sign.ipynb  # train and test
├── asl_model.h5     # trained model
│   
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the Repository

```
git clone https://github.com/sivasankarim-41/asl-language-detection.git
cd asl-language-detection
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Project

```
python src/sign.ipynb

```

👉 Make sure your webcam is enabled.

---

## 🧠 Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 64 × 64 RGB images
* Output: 29 classes (A–Z + special symbols)

### Architecture:

* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Flatten → Dense (128 neurons)
* Output Layer (Softmax)

---

## 📊 Dataset

* ASL Alphabet Dataset
* Contains labeled images of hand gestures for each alphabet
* Used ImageDataGenerator for preprocessing and validation split

---



## 🔮 Future Improvements

* Convert detected letters into full words/sentences
* Improve model accuracy using advanced architectures
* Add confidence score display
* Deploy as a web application using Streamlit
* Add voice output for accessibility

---

## ⚠️ Notes

* If the model file (`asl_model.h5`) is large, it may not be uploaded to GitHub
* You can retrain the model using `train_model.py`

---

## 👨‍💻 Author

**Your Name**
GitHub: https://github.com/sivasankarim-41


---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
