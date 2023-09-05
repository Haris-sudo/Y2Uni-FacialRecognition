
import cv2
import random
import image_utils
import numpy as np
from os import path, mkdir, listdir

class Train:

    def __init__ (self):
        self.cam = cv2.VideoCapture(0)

    def __del__(self):
        self.cam.release()

    #creating student directories based on supplied studentID
    def create_student_dir(self, student_id):
        user_data_dir = "user_data"
        if (not path.exists(user_data_dir)):
            mkdir(user_data_dir)
        user_dir = user_data_dir + "/" + student_id
        if (not path.exists(user_dir)):
            mkdir(user_dir)
        return user_dir

    def run_training(self):
        user_data_dir = "user_data"
        #finding user image directories
        user_dirs = listdir(user_data_dir)
        images = []
        ids = []
        #iterate the student IDs
        for u in user_dirs:
            user_files = listdir(user_data_dir + "/" + u)
            #iterating student images for IDs
            for f in user_files:
                image = cv2.imread(user_data_dir + "/" + u + "/" + f)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ids.append(int(u))
                images.append(image)
        recogniser = cv2.face.LBPHFaceRecognizer_create()
        recogniser.train(images, np.array(ids))
        recogniser.write("student_train.yml")

    #generate training set of student images, takes 30 images of the student when the face is detected
    def generate_training_set(self, student_id):
        student_dir = self.create_student_dir(student_id)
        number_of_images = 30
        saved_images = 0
        current_frame = 0

        while saved_images < number_of_images:
            ret, image = self.cam.read()
            if(current_frame % 10 == 0):
                if(not ret):
                    return False
                image, bounding_box = image_utils.crop_and_greyscale(image)
                if(not image.size == 0):
                    cv2.imwrite(student_dir + "/" + str(random.randint(0,50000))+ ".jpg", image)
                    print ("image saved")
                    saved_images += 1
            current_frame += 1

r = Train()
r.generate_training_set("93")
#r.run_training()


    #def recognise_from_video():
    #    recogniser = cv2.face.LBPHFaceRecognizer_create()
    #    recogniser.read("student_train.yml") 
    #    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #    cap = cv2.VideoCapture(0)

        #while True:
         #   ret, frame = cap.read()
          #  faces = face_cascade.detectMultiScale(frame, 1.3, 5)
           # for (x, y, w, h) in faces:
            #    image_grey = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
             #   image_grey = cv2.resize(image_grey, (200, 200))
              #  id, confidence = recogniser.predict(image_grey)
               # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                #student_info = "Student ID: " + str(id) + " (" + str(confidence) + "%)"
            #display text
                #if (confidence > 0):
                 #   cv2.putText(frame, student_info, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # display
            #cv2.imshow('frame', frame)

            # esc to stop
            #k = cv2.waitKey(30) & 0xff
            #if k==27:
             #   break
        #cap.release()
        #cv2.destroyAllWindows()

    #recognise_from_video()
    #cv2.destroyAllWindows()




