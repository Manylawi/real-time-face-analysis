import cv2
import numpy as np

GENDER_MODEL = 'gender_net.caffemodel'
GENDER_PROTO = 'gender_deploy.prototxt'

gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

GENDER_LIST = ['Male', 'Female']

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# نموذج التعرف على الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

        # تقدير الجنس
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        label = f"{gender}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gender Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()