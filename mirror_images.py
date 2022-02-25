import glob
import os
import cv2
import numpy as np
import matplotlib as plt


dataset_path = './Obj_Dataset/Data/'
os.chdir(dataset_path)
for image in glob.glob("*.jpg"):
    img = cv2.imread(image)
    temp_name = image.split('_')
    image_mirror = np.fliplr(img)
    filename = temp_name[0] + '_' + temp_name[1] + '_' + temp_name[2][:-4] + "_flip.jpg"
    cv2.imwrite(filename, image_mirror)