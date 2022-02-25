import copy
import glob
import os
import cv2
import mediapipe as mp
from app_files import calc_landmark_list, pre_process_landmark, logging_csv

letter_dict = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7, 'i' : 8,
               'j' : 9, 'k' : 10, 'l' : 11, 'm' : 12, 'n' : 13, 'o' : 14, 'p' : 15, 'q' : 16,
               'r' : 17, 's' : 18, 't' : 19, 'u' : 20, 'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24, 'z' : 25, 'Space':26}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
path = 'Obj_Dataset/Data/'
original_path = os.getcwd()
IMAGE_FILES = [img for img in glob.glob(path + '*.jpg')]

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4, model_complexity=0) as hands:
  count = 0
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).

    image = cv2.flip(cv2.imread(file), 1)
    debug_image = copy.deepcopy(image)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.

    if not results.multi_hand_landmarks:

        count += 1
        continue


    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())



    save_path = '/home/angel/PycharmProjects/ASL_recognition/Annotated'
    os.chdir(save_path)
    letter = file.split('/')[-1].split('_')[0]
    tittle = str(letter) + '_'+ str(idx) + '_.jpg'
    print(cv2.imwrite(tittle, cv2.flip(annotated_image, 1)))

    os.chdir(original_path)


    for hand_landmarks in results.multi_hand_landmarks:
        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        logging_csv(letter_dict[letter], pre_processed_landmark_list)




    # Draw hand world landmarks.
    # if not results.multi_hand_world_landmarks:
    #   continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    #   mp_drawing.plot_landmarks(
    #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
  print(count)