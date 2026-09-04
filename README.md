# 🚨 Smart Traffic Sign Recognition & Alert System

An end-to-end **Machine Learning application for traffic sign recognition and warning** using an **Artificial Neural Network (ANN)**. The system can recognize traffic signs from **images, videos, and live camera input**, display the predicted sign with its confidence score, and provide a **voice alert** for the detected traffic sign.

## 📌 Project Overview

Traffic sign recognition is an important application of **Computer Vision and Machine Learning**, with applications in intelligent transportation systems and driver-assistance systems.

Traffic signs provide essential information about speed limits, road conditions, warnings, restrictions, and other traffic regulations. Automatically recognizing these signs can improve road safety and support intelligent vehicle systems by providing timely alerts to drivers.

This project uses a trained **Artificial Neural Network (ANN)** to recognize and classify traffic signs. The application is developed using **Python, TensorFlow/Keras, OpenCV, and Streamlit**.

## 🎯 Objectives

* Develop an **ANN-based traffic sign recognition and warning system**.
* Enable users to provide **images, videos, and live camera input**.
* Preprocess and normalize traffic sign images for accurate prediction.
* Classify traffic signs using a trained **Artificial Neural Network (ANN)**.
* Display the **predicted traffic sign and confidence score**.
* Provide a **voice alert** for the detected traffic sign.
* Build an interactive **Streamlit application** for traffic sign recognition and warning.

## 📊 Dataset

The Traffic Sign Dataset is an image classification dataset designed for recognizing and classifying different types of traffic signs.

The dataset contains images organized into different classes, where each class represents a specific traffic sign. The images are preprocessed and resized to **32 × 32 pixels** before being provided to the ANN model.

### Dataset Features

* Traffic sign images
* Multiple traffic sign classes
* Image size: **32 × 32 pixels**
* RGB/BGR image data
* Flattened pixel features for ANN training

## 🧠 Machine Learning Model

The project uses an **Artificial Neural Network (ANN)** for traffic sign classification.

### Hyperparameter Optimization

The ANN hyperparameters were optimized by testing different configurations:

* **Number of Layers:** Tested different hidden-layer architectures.
* **Neurons:** Evaluated different numbers of neurons in the hidden layers.
* **Dropout:** Tested different dropout rates to reduce overfitting.
* **Learning Rate:** Evaluated different learning-rate values.
* **Batch Size:** Tested different batch sizes.
* **Random Search:** Explored different combinations of ANN hyperparameters.
* **Final Training:** Trained the final ANN using the best-performing configuration.

### Best Configuration

* **Hidden Layers:** 3
* **Neurons:** 160, 192, 160
* **Dropout:** 0.1 and 0.3
* **Learning Rate:** 0.000321
* **Batch Size:** 16
* **Validation Accuracy:** **75.41%**

## ⚙️ Image Processing

The input traffic sign image is processed before prediction:

1. Capture or upload the input image.
2. Resize the image to **32 × 32 pixels**.
3. Convert pixel values to floating-point values.
4. Normalize pixel values using `/255.0`.
5. Flatten the image into a feature vector.
6. Pass the processed input to the trained ANN model.
7. Predict the traffic sign class.
8. Display the predicted class and confidence score.

## 🚦 Application Features

### 📷 Image Prediction

Upload a traffic sign image and get the predicted traffic sign and confidence score.

### 🎥 Video Prediction

Upload a traffic sign video and process frames to identify traffic signs.

### 📹 Live Camera

Use a live camera feed for real-time traffic sign recognition.

### 🔊 Voice Alert

The application provides a voice alert for the detected traffic sign, such as:

> "Predicted traffic sign is No Parking."

### 📊 Confidence Score

The application displays the model's prediction confidence for the detected traffic sign.

## 🖥️ Technology Stack

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| Python             | Programming language        |
| TensorFlow / Keras | ANN model development       |
| NumPy              | Numerical operations        |
| Pandas             | Dataset processing          |
| OpenCV             | Image and video processing  |
| Scikit-learn       | Machine learning utilities  |
| Streamlit          | Web application             |
| MediaPipe          | Real-time vision processing |
| Matplotlib         | Data visualization          |
| Seaborn            | Data visualization          |
| Joblib             | Model/data utilities        |

## 📁 Project Structure

```text
TrafficSign-AI-Traffic-Sign-Alert-System/
│
├── app.py
│
├── data/
│   └── traffic_sign.csv
│
├── models/
│   └── vehicle_sign_ann.keras
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PYARASANISRINIVAS/TrafficSign-AI-Traffic-Sign-Alert-System.git
```

### 2. Navigate to the Project

```bash
cd TrafficSign-AI-Traffic-Sign-Alert-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Application

**Traffic Sign Recognition & Alert System**

https://traffic-sign-alert-system.streamlit.app/

## 🔄 System Workflow

```text
Input
  │
  ├── Image
  ├── Video
  └── Live Camera
       │
       ▼
Image / Frame Capture
       │
       ▼
Resize to 32 × 32
       │
       ▼
Pixel Normalization
       │
       ▼
Flatten Image
       │
       ▼
Trained ANN Model
       │
       ▼
Traffic Sign Prediction
       │
       ├── Predicted Sign
       ├── Confidence Score
       └── Voice Alert
```

## 📈 Result

The optimized ANN configuration achieved a **75.41% validation accuracy**.

The system successfully provides traffic sign predictions through image, video, and live camera input and presents the detected sign with a confidence score and voice alert.

## 🔮 Future Enhancements

* Improve model accuracy using a larger and more diverse dataset.
* Experiment with CNN-based architectures for image classification.
* Add more traffic sign categories.
* Improve real-time detection performance.
* Integrate object detection for detecting signs within complex road scenes.
* Add multilingual voice alerts.
* Deploy the system for intelligent vehicle and driver-assistance applications.

## 👨‍💻 Author

**Pyarasani Srinivas**

B.Tech – Computer Science and Engineering
CMR Engineering College, JNTUH

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
