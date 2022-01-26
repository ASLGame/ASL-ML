import cv2
from cvzone.HandTrackingModule import HandDetector
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands = 1)
predictions = []
padding = 50
alphabet = 'abcdefghijklmnopqrstuvwxyz-;'
directory = os.getcwd() + '/Dataset'

def SaveImage(key, hand):
    letter = chr(key)
    if letter == '-':
        letter = "Space"
    if letter == ';':
        letter = 'Delete'

    print('Saving the letter: ', letter)
    if os.getcwd() != directory + f'/{letter}':
        os.chdir(directory + f'/{letter}')
    count = len([name for name in os.listdir('.') if os.path.isfile(name)])
    filename = f'{letter}_{count}.jpg'
    cv2.imwrite(filename, hand)




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
        if chr(key) in alphabet:
            SaveImage(key, hand)

        # if key == ord('a'):
        #     print(os.getcwd())
        #     if os.getcwd() != directory:
        #         os.chdir(directory)
        #
        #     count = len([name for name in os.listdir('.') if os.path.isfile(name)])
        #     filename = f'a_{count}.jpg'
        #     cv2.imwrite(filename, hand)

    cv2.imshow('MainWin', img)
    close = cv2.waitKey(1) & 0xFF

    if close == ord("."):
        break
