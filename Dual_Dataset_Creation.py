import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import json
from datetime import date

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands = 1)
predictions = []
padding = 50
alphabet = 'abcdefghijklmnopqrstuvwxyz-;'
classification_directory = os.getcwd() + '/Classification_Dataset'
obj_directory = os.getcwd() + '/Obj_Dataset'
f = open(obj_directory + '/Annotations.json')
Annotations = json.load(f)
alphabet_dict = {}



for i, c in enumerate(alphabet):
    alphabet_dict.setdefault(c, i)

def SaveImageClassification(key, hand):
    letter = chr(key)
    if letter == '-':
        letter = "Space"
    if letter == ';':
        letter = 'Delete'

    print('Saving the letter: ', letter)
    if os.getcwd() != classification_directory + f'/{letter}':
        os.chdir(classification_directory + f'/{letter}')
    count = len([name for name in os.listdir('.') if os.path.isfile(name)])
    filename = f'{letter}_{count}.jpg'
    cv2.imwrite(filename, hand)

    return [os.getcwd(), letter, filename]

def SaveImageObjDetect(key, img, hand, bbox):
    letter = chr(key)
    if letter == '-':
        letter = "Space"
    if letter == ';':
        letter = 'Delete'

    print("Saving to Obj detection dataset...")
    if os.getcwd() != obj_directory + '/Data':
        os.chdir(obj_directory + '/Data')

    count = len([name for name in os.listdir('.') if os.path.isfile(name)])
    (h, w)  = hand.shape[:2]
    filename = f'{letter}_{count}.jpg'
    cv2.imwrite(filename, img)
    img_dict = {
        "license" : 1,
        "filename" : filename,
        "width": w,
        "height": h,
        "id": count
    }
    print("     Adding image to data folder")
    Annotations['images'].append(img_dict)

    annotation = {
        "image_id" : count,
        "id" : len(Annotations['annotations']),
        "bbox" : bbox,
        "category_id" : alphabet_dict[letter]

    }
    print("     Adding annotation...")
    Annotations['annotations'].append(annotation)

    return [os.getcwd(), filename]




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
            obj_undo = SaveImageObjDetect(key, img, hand, bbox)
            classification_undo = SaveImageClassification(key, hand)


    cv2.imshow('MainWin', img)
    close = cv2.waitKey(1) & 0xFF
    if close == ord('/'):
        # REMOVE LAST ITEM FROM THE OBJECT DETECTION DATASET
        if os.getcwd() != obj_directory + '/Data':
            os.chdir(obj_undo[0])
        os.remove(obj_undo[1])
        print(f'IN OBJ DATASET REMOVED FROM {obj_undo[0]} THE FILE {obj_undo[1]}')
        Annotations['images'].pop()
        Annotations['annotations'].pop()

        # REMOVE LAST ITEM FROM THE CLASSIFICATION DATASET
        if os.getcwd() != classification_directory + f'/{classification_undo[1]}':
            os.chdir(classification_directory + f'/{classification_undo[1]}')
        os.remove(classification_undo[2])

        print(f'IN CLASSIFICATION DATASET REMOVED FROM {classification_undo[0]} THE FILE {classification_undo[2]}')

    if close == ord("."):
        json_file = json.dumps(Annotations, indent=4)
        os.chdir(obj_directory)
        with open("Annotations.json", "w") as outfile:
            outfile.write(json_file)
        break
