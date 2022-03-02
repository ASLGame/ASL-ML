import copy

import cv2
import time
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from app_files import calc_landmark_list, pre_process_landmark, draw_info_text

from WebCamStream import WebcamStream

# initializing and starting multi-threaded webcam capture input stream
webcam_stream = WebcamStream()
webcam_stream.start()

ASL_model = load_model('keypoint_classifier_final.h5')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h',
          8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p',
            16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x',
            24:'y', 25:'z', 26:"space"}

hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


prev_frame_time = 0


new_frame_time = 0

num_frames_processed = 0
start = time.time()
while True:
    new_frame_time = time.time()
    if webcam_stream.stopped is True:
        break

    else:
        image = webcam_stream.read()

    num_frames_processed += 1

    debug_image = copy.deepcopy(image)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(cv2.flip(image, 1))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = np.array(pre_process_landmark(landmark_list))
            pre_processed_landmark_list = np.array([pre_processed_landmark_list], dtype=np.float32)
            key = cv2.waitKey(1) & 0xFF
            # if key == ord('p'):
            prediction = ASL_model.predict(pre_processed_landmark_list)

            letter = np.argmax(prediction)

            debug_image = draw_info_text(
                debug_image,
                alphabet[letter])

    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(debug_image, str(fps), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('MainWin', debug_image)
    close = cv2.waitKey(1) & 0xFF

    if close == ord("q"):
        break


end = time.time()
webcam_stream.stop()

# printing time elapsed and fps
elapsed = end-start
fps = num_frames_processed/elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows
cv2.destroyAllWindows()