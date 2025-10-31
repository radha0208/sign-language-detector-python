# 🤟 Sign Language Detection System

A **real-time Sign Language Detection** system built using **Python, OpenCV, and Mediapipe**, designed to recognize and translate hand gestures into text and speech.

---

## 🧠 Features

- 🖐️ Detects and tracks hand gestures in real-time using **Mediapipe**
- 🤖 Classifies gestures using a trained **Machine Learning model (Scikit-learn)**
- 🔡 Supports **Alphabet** and **Word** detection modes
- 🎨 Interactive **Tkinter GUI** for a user-friendly experience
- 🌐 **Translation** support using Google Translate API
- 🔊 **Text-to-Speech** via gTTS (Google Text-to-Speech)

---

## 🧰 Tech Stack

| Component        | Technology Used         |
|------------------|--------------------------|
| GUI              | Tkinter                  |
| Computer Vision  | OpenCV, Mediapipe        |
| ML Model         | Scikit-learn             |
| Data Handling    | NumPy                    |
| Visualization    | Matplotlib               |
| Translation      | Googletrans              |
| Speech Output    | gTTS, Playsound          |
| Image Handling   | Pillow                   |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/radha0208/sign-language-detector-python.git
cd sign-language-detector-python
```
### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
python collect_imgs.py         # Collects real-time images through webcam
python create_dataset.py       # Creates dataset by extracting Mediapipe landmarks
python train_classifier.py     # Trains the MLP model
python detect.py               # Detects gestures and translates them in real-time
```
### 📊 How It Works

Hand Detection → Mediapipe detects hand landmarks from the live webcam feed.<br>
Feature Extraction → Landmark coordinates are processed and scaled.<br>
Classification → The trained ML model predicts the corresponding gesture.<br>
Translation (optional) → The gesture can be translated using Google Translate.<br>
Speech Output → The predicted text is spoken using gTTS and playsound.

### 🖼️ Sample Outputs
Here are some sample outputs from the Sign Language Detection System 👇

<p align="center"> <img src="assets/Screenshot 2025-10-30 155738.png" width="45%"> <br><img src="assets/Screenshot 2025-10-30 155421.png" width="45%"> </p> <p align="center"> <img src="assets/Screenshot 2025-10-30 155352.png" width="45%"><br> <img src="assets/Screenshot 2025-10-30 155241.png" width="45%"> </p> <p align="center"> <img src="assets/Screenshot 2025-10-30 155224.png" width="45%"> </p>

### 👩‍💻 Author
Radha (@Radha0208)


