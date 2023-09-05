import cv2
import image_utils
import base64
import tensorflow as tf
import numpy as np

class MaskRecogniser:
    def __init__ (self):
        #initialsing and loading training files
        self.cam = None
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")
        self.active = True

        #self.model = tf.keras.models.load_model("mask_model")

    def __del__(self):
        self.cam.release()
    
    def set_camera(self, camera):
        self.cam = camera

    #detecting mask
    def get_mask(self):
        ret, image = self.cam.read()
        if(not ret):
            print("FAILED TO GET IMAGE")
        image_color, bounding_box = image_utils.crop_colour(image)
        if(not image_color.size == 0):
            rsize_image = cv2.resize(image_color, (200, 200))
            np_arr = np.array(rsize_image)
            np_arr = np.expand_dims(np_arr, axis = 0)
            prediction = self.model.predict(np_arr/255)

            #box for face detection
            (x, y, w, h) = bounding_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            student_info = "Mask: " + str(prediction)
            #masked - threshold for prediction if mask is on student
            if(prediction > 0.9999):
                cv2.putText(image, student_info, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #unmasked
            else:  
                cv2.putText(image, student_info, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            ret, image2 = cv2.imencode('.jpg', image)
            data = base64.b64encode(image2).decode("UTF-8")
            return -1, prediction[0][0], data
        else:
            ret, image2 = cv2.imencode('.jpg', image)
            data = base64.b64encode(image2).decode("UTF-8")
            return -1, -1, data


class Recogniser:
    #initialsing and loading training file
    def __init__ (self):
        self.cam = None
        self.recogniser = cv2.face.LBPHFaceRecognizer_create()
        self.recogniser.read("student_train.yml")
        self.active = True

    def __del__(self):
        self.cam.release()

    def set_camera(self, camera):
        self.cam = camera

    def get_student_id(self):
        ret, image = self.cam.read()
        if(not ret):
            print("FAILED TO GET IMAGE")
        image_grey, bounding_box = image_utils.crop_and_greyscale(image)
        if(not image_grey.size == 0):
            studentID, confidence = self.recogniser.predict(image_grey)
            #is_mask, confidence = self.mask_recogniser.predict(image_grey)

            #box for face detection
            (x, y, w, h) = bounding_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            student_info = "Student ID: " + str(studentID) + " (" + str(confidence) + "%)"
            cv2.putText(image, student_info, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ret, image = cv2.imencode('.jpg', image)
            data = base64.b64encode(image).decode("UTF-8")
            return studentID, confidence, data
        else:
            ret, image = cv2.imencode('.jpg', image)
            data = base64.b64encode(image).decode("UTF-8")
            return -1, -1, data

#r = MaskRecogniser()

#while True:
#   s, c, i = r.get_student_id()
#   cv2.imshow('video', i)
#   k = cv2.waitKey(30) & 0xff
#   if k == 27: # press 'ESC' to quit
#       break

cv2.destroyAllWindows()
