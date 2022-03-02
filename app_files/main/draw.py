import cv2 as cv

def draw_info_text(image,  hand_sign_text):

    #info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = "Predicted Text" + ':' + hand_sign_text

    cv.putText(image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    return image