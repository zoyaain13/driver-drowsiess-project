
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import winsound

# Load trained model
model = tf.keras.models.load_model("drowsiness_model.keras")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face detector could not be loaded.")

# Settings
CLOSED_THRESHOLD = 0.35
CLOSED_SECONDS = 2.5

closed_start = None
alarm_active = False
last_alarm = 0

def play_alarm():
    global alarm_active
    try:
        winsound.Beep(1200, 1000)
    finally:
        alarm_active = False

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

print("Driver Drowsiness Detection Started")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(80, 80)
    )

    status = "NO FACE DETECTED"
    color = (0, 0, 255)

    if len(faces) > 0:

        # Select largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Approximate eye regions in upper face
        face_gray = gray[y:y+h, x:x+w]

        eye_regions = [
            (0.12, 0.22, 0.48, 0.53),
            (0.52, 0.22, 0.88, 0.53)
        ]

        probabilities = []

        for index, (x1, y1, x2, y2) in enumerate(eye_regions):

            ex1 = int(w * x1)
            ey1 = int(h * y1)
            ex2 = int(w * x2)
            ey2 = int(h * y2)

            eye_img = face_gray[ey1:ey2, ex1:ex2]

            if eye_img.size == 0:
                continue

            eye_img = cv2.resize(eye_img, (64, 64))
            eye_img = eye_img.astype("float32")
            eye_img = np.expand_dims(eye_img, axis=(0, -1))

            prediction = float(
                model.predict(eye_img, verbose=0)[0][0]
            )

            probabilities.append(prediction)

            cv2.rectangle(
                frame,
                (x+ex1, y+ey1),
                (x+ex2, y+ey2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Eye {index+1}: {prediction:.2f}",
                (x+ex1, y+ey1-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        if probabilities:

            avg_open = float(np.mean(probabilities))

            cv2.putText(
                frame,
                f"Open probability: {avg_open:.2f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if avg_open < CLOSED_THRESHOLD:

                if closed_start is None:
                    closed_start = time.time()

                duration = time.time() - closed_start

                status = "EYES CLOSED"
                color = (0, 0, 255)

                if duration >= CLOSED_SECONDS:

                    status = "DROWSINESS DETECTED!"
                    color = (0, 0, 255)

                    if not alarm_active and time.time() - last_alarm > 1:
                        alarm_active = True
                        last_alarm = time.time()

                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

            else:
                closed_start = None
                status = "EYES OPEN"
                color = (0, 255, 0)

        else:
            status = "EYE REGION NOT AVAILABLE"
            color = (0, 165, 255)
            closed_start = None

    else:
        closed_start = None

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Driver Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("System stopped.")
