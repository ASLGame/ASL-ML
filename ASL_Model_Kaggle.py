import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands = 1)
ASL_model = load_model('hand_sign_classification.h5')
predictions = []
padding = 50
alphabet = {1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h',
          9:'i', 10:'j', 11:'k', 12:'l', 13:'m', 14:'n', 15:'o', 16:'p',
            17:'q', 18:'r', 19:'s', 20:'t', 21:'u', 22:'v', 23:'w', 24:'x',
            25:'y', 26:'z', 27:'del', 28:'nothing', 29:"space"}


while True:
    succ, img = cap.read()
    (h, w) = img.shape[:2]
    hands= detector.findHands(img, draw=False)

    if hands:
        bbox = hands[0]['bbox']
        x, y, boxW, boxH = bbox[0], bbox[1], bbox[2], bbox[3]
        startX, startY = max(0, x), max(0, y)
        endX, endY = min(w-1, x + boxW), min(h-1, y + boxH)
        if startX < padding and startY < padding:
            hand = img[startY: endY + padding, startX : endX + padding]
        elif startX < padding:
            hand = img[startY - padding: endY + padding, startX: endX + padding]
        elif startY < padding:
            hand = img[startY : endY + padding, startX - padding: endX + padding]
        else:
            hand = img[startY - padding: endY + padding, startX - padding: endX + padding]

        cv2.imshow("Hand", hand)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            hand = cv2.resize(hand, (64, 64))
            hand = img_to_array(hand)
            hand = preprocess_input(hand)
            hand = np.expand_dims(hand, axis=0)

            prediction = ASL_model.predict(hand)[0]

            hand_sign = np.argmax(prediction)
            print(alphabet[hand_sign+1])
    cv2.imshow('MainWin', img)
    close = cv2.waitKey(1) & 0xFF

    if close == ord("q"):
        break
