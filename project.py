import os
import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load models
def load_all_models():
    embedder = FaceNet()
    emotion_model = load_model("face_model.h5")
    emotion_labels = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
    gender_labels = ['Male', 'Female']
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    knn = joblib.load("knn_model.pkl")
    le = joblib.load("label_encoder.pkl")

    return {
        "embedder": embedder,
        "emotion_model": emotion_model,
        "emotion_labels": emotion_labels,
        "gender_net": gender_net,
        "gender_labels": gender_labels,
        "face_detection": face_detection,
        "face_cascade": face_cascade,
        "knn": knn,
        "le": le
    }

# Draw shadowed text
def draw_text(img, text, pos, scale=0.6, color=(255, 255, 255), shadow=(0, 0, 0)):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, scale, shadow, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, 1, lineType=cv2.LINE_AA)

# Main loop
def main():
    models = load_all_models()
    cap = cv2.VideoCapture(0)
    print("Starting face analysis system. Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        counts = {"total": 0, "male": 0, "female": 0, "known": 0, "unknown": 0}
        results = models["face_detection"].process(rgb)

        if results.detections:
            counts["total"] = len(results.detections)
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x1, y1 = int(box.xmin * w), int(box.ymin * h)
                x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                try:
                    # Face recognition
                    face160 = cv2.resize(face_img, (160, 160))
                    emb = models["embedder"].embeddings([face160])[0]
                    name, dist = "Unknown", 1.0
                    if models["knn"] and models["le"]:
                        dists, _ = models["knn"].kneighbors([emb], n_neighbors=1)
                        dist = dists[0][0]
                        if dist < 0.9:
                            name = models["le"].inverse_transform(models["knn"].predict([emb]))[0]
                            counts["known"] += 1
                        else:
                            counts["unknown"] += 1
                    else:
                        counts["unknown"] += 1

                    # Emotion detection
                    emotion_face = cv2.resize(face_img, (48, 48))
                    emotion_face = cv2.cvtColor(emotion_face, cv2.COLOR_BGR2GRAY)
                    emotion_face = image.img_to_array(emotion_face)
                    emotion_pred = models["emotion_model"].predict(np.expand_dims(emotion_face, axis=0), verbose=0)
                    emotion_idx = np.argmax(emotion_pred)
                    emotion = models["emotion_labels"][emotion_idx]
                    emotion_conf = float(emotion_pred[0][emotion_idx])

                    # Gender detection
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
                    models["gender_net"].setInput(blob)
                    gender_pred = models["gender_net"].forward()[0]
                    gender_idx = gender_pred.argmax()
                    gender = models["gender_labels"][gender_idx]
                    gender_conf = float(gender_pred[gender_idx])
                    counts[gender.lower()] += 1

                    # Draw face box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display info (smaller text above each face)
                    y_offset = y1 - 8
                    draw_text(frame, f"ID: {name} ({dist:.2f})", (x1, y_offset), scale=0.45)
                    y_offset -= 16
                    draw_text(frame, f"Emotion: {emotion} ({emotion_conf:.2f})", (x1, y_offset), scale=0.45)
                    y_offset -= 16
                    draw_text(frame, f"Gender: {gender} ({gender_conf:.2f})", (x1, y_offset), scale=0.45)

                except Exception as e:
                    print("Error:", e)
                    continue

        # Display horizontal counters at top
        horizontal_counters = (
            f'Total: {counts["total"]}   '
            f'Male: {counts["male"]}   '
            f'Female: {counts["female"]}   '
            f'Known: {counts["known"]}   '
            f'Unknown: {counts["unknown"]}'
        )
        draw_text(frame, horizontal_counters, (10, 25), scale=0.6, color=(255, 255, 0))

        cv2.imshow("Face Analysis System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System stopped.")

if __name__ == "__main__":
    main()