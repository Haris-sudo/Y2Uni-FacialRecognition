import cv2
import image_utils
import base64
import numpy as np
import os
import random

class Sampler:
    def __init__ (self):
        self.cam = cv2.VideoCapture(0)

    def __del__(self):
        self.cam.release()
    #creating temp directories for dataset
    def create_dirs(self):
        os.mkdir("temp_data")
        os.mkdir("temp_data/masked")
        os.mkdir("temp_data/unmasked")
    #saving images taken 
    def save_image(self, im, masked = False):
        number = random.randint(0, 100000)
        if(masked):
            cv2.imwrite("temp_data/masked/" + str(number) + ".jpg", im)
        else:
            cv2.imwrite("temp_data/unmasked/" + str(number) + ".jpg", im)
        return

    #detecting face then displaying box around detected face
    def get_image(self):
        ret, image = self.cam.read()
        if(not ret):
            print("FAILED TO GET IMAGE")
        image_color, bounding_box = image_utils.crop_colour(image)
        if(not image_color.size == 0):
            rsize_image = cv2.resize(image_color, (200, 200))
            (x, y, w, h) = bounding_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return 1, rsize_image, image
        else:
            return -1, None, image

r = Sampler()
r.create_dirs()
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#controls to take images after detecting face (only when box (detected face) is present)
while True:
    success, r_image, i = r.get_image()
    cv2.imshow('video', i)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    if k == 97 and success == 1: # press 'A' to save unmasked
        print("Saving Unmasked Image")
        r.save_image(r_image, False)
    elif k == 98 and success == 1: # press 'B' to save masked
        print("Saving Masked Image")
        r.save_image(r_image, True)
    #no face detected whilst pressing A or B (no box around face) - nothing saved
    elif k == 97 or k == 98:
        print("Failed To Find Face")