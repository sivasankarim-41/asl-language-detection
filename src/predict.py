import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

model = tf.keras.models.load_model("models/asl_model.h5")

labels = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","space","nothing"
]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        x, y, w, h = hands[0]['bbox']

        y1, y2 = max(0,y-offset), min(img.shape[0], y+h+offset)
        x1, x2 = max(0,x-offset), min(img.shape[1], x+w+offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            imgResize = cv2.resize(imgCrop, (64, 64)) / 255.0
            imgResize = np.expand_dims(imgResize, axis=0)

            prediction = model.predict(imgResize)
            idx = np.argmax(prediction)
            letter = labels[idx]

            cv2.putText(img, letter, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)

    cv2.imshow("ASL Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()