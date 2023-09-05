import cv2
import numpy as np
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"


def crop_and_greyscale(image):
    cascade_class = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    #change to greyscale
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detecting faces
    faces = cascade_class.detectMultiScale(image_grey, 1.3, 5)

    if(len(faces) > 0):
        bounding_box = faces[0]
        (x, y, w, h) = bounding_box
        #crops
        cropped_image = image_grey[y: y + h, x: x + w]
        #resize
        cropped_image = cv2.resize(cropped_image, (200, 200))
        return cropped_image, bounding_box
    
    return np.array([]), None

def crop_colour(image):
    cascade_class = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    #change to greyscale
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detecting faces
    faces = cascade_class.detectMultiScale(image_grey, 1.3, 5)

    if(len(faces) > 0):
        bounding_box = faces[0]
        (x, y, w, h) = bounding_box
        #crops
        cropped_image = image[y: y + h, x: x + w]
        #resize
        cropped_image = cv2.resize(cropped_image, (200, 200))
        return cropped_image, bounding_box
    
    return np.array([]), None