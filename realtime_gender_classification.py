import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

# Load trained model
model = load_model("my_model.h5")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    
    tf.config.set_visible_devices(gpus[0], 'GPU')
    
    # Restrict TensorFlow to only allocate memory on the first GPU
    tf.config.experimental.set_memory_growth(gpus[0], True)
# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for x, y, w, h in faces:
        face_img = frame[y : y + h, x : x + w]
        face_resized = cv2.resize(face_img, (64, 64))
        img_array = face_resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        if prediction[0][0] >= 0.5:
            label = f"Female ({prediction[0][0]*100:.1f}%)"
            color = (0, 0, 255)
        else:
            label = f"Male ({(1-prediction[0][0])*100:.1f}%)"
            color = (255, 0, 0)
            # Draw border if detected as male
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "MONITOR",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame, label, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("Gender Classification - Real Time", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
