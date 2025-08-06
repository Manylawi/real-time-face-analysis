# 🤖 Real-Time AI Face Analysis System

A real-time facial analysis system that performs **face detection**, **recognition**, **emotion detection**, and **gender classification** using computer vision and deep learning — all from a webcam feed.

---

## 🎓 Final Project — National Telecommunication Institute (NTI)  
Faculty of Engineering, Helwan National University – 2025  
**Supervised by**: Dr. Mohamed Zorkany

---

## 👨‍💻 Our Team

- Ahmed Mohamed El-Manylawi  
- Ahmed Saeed Mohamed
- Ahmed Reda Kamel  
- Youssef Ahmed  
- Ibrahim Hussein  
- Ahmed Essam

---

## 🧠 Project Overview

This system uses a live webcam stream to:
- Detect human faces in real time
- Identify known individuals using FaceNet + KNN
- Predict facial emotions (Happy, Sad, Angry, etc.)
- Classify gender as Male or Female
- Show real-time counters: total faces, known/unknown, male/female
- Visualize all data live with bounding boxes and overlays

### 🧩 Applications
- Smart attendance systems  
- Sentiment analysis in crowds  
- Security & access control  
- Behavior tracking in smart classrooms  

---

## 🛠️ Tech Stack & Components

| Module         | Purpose                                         |
|----------------|--------------------------------------------------|
| `OpenCV`       | Image capture, face box drawing, processing      |
| `MediaPipe`    | Real-time face detection                         |
| `Keras-FaceNet`| Face embedding generation (128-d vectors)        |
| `KNN (Joblib)` | Face recognition (known vs unknown)              |
| `TensorFlow`   | Emotion classification (CNN model)              |
| `Caffe`        | Gender classification using pre-trained model    |
| `cv2.CascadeClassifier` | Backup face detection method           |

---

## 🔍 Core Features

- ✅ Face Detection using MediaPipe  
- 🧑‍🦰 Face Recognition using FaceNet + KNN  
- 😊 Emotion Detection (7 categories)  
- 👨‍🦰 Gender Detection via Caffe DNN  
- 🔢 Real-time Counters (known/unknown, male/female, total)  
- 📺 Live webcam overlay & visual feedback  

---

## 📂 File Structure

```
face_analysis.py             → Main app code
face_model.h5                → Pre-trained CNN emotion model
knn_model.pkl                → Trained KNN model for face recognition
label_encoder.pkl            → Label encoder for person names
gender_net.caffemodel        → Gender classification weights
gender_deploy.prototxt       → Gender model structure
requirements.txt             → Required Python packages
```

---

## ▶️ How to Run

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Make sure the following files exist** in the same folder:
- `face_model.h5`
- `knn_model.pkl`
- `label_encoder.pkl`
- `gender_net.caffemodel`
- `gender_deploy.prototxt`

3. **Run the system**:
```bash
python face_analysis.py
```

4. **Press `q` to exit.**

---

## 🙌 Acknowledgments

Special thanks to:
- 🧑‍🏫 **Dr. Mohamed Zorkany** — Project Supervisor  
- 🏫 **NTI – National Telecommunication Institute**  
For supporting and mentoring us throughout the development of this smart vision system.

---

## 📜 License

This project is for **educational and academic demonstration only**.  
All rights reserved © 2025 by the project team.
