import tensorflow as tf
import cv2
import random
import image_utils
import numpy as np
from os import path, mkdir, listdir
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#loading images of mask/unmasked dataset with validation images
class MaskTrainNN:
    def load_training_set2(self):
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        test_datagen = ImageDataGenerator(rescale=1./255)

        #training set
        train_generator = train_datagen.flow_from_directory(
            'temp_data/train', 
            target_size=(200, 200), 
            batch_size=4,
            class_mode='binary')  

        #validation set
        validation_generator = test_datagen.flow_from_directory(
            'temp_data/validation',
            target_size=(200, 200),
            batch_size=4,
            class_mode='binary')

        return train_generator, validation_generator


    def load_training_set(self):
        print("Loading Training Set")
        mask_data_dir = "mask_data"
        masked_folder = "masked"
        unmasked_folder = "unmasked"

        #finding mask dataset directories
        mask_dirs = listdir(mask_data_dir + "/" + masked_folder)
        unmask_dirs = listdir(mask_data_dir + "/" + unmasked_folder)
        images = []
        ids = []
        masked_found = 0
        unmasked_found = 0

        #iterate the masked data set
        for u in mask_dirs:
            user_files = listdir(mask_data_dir + "/" + masked_folder + "/" + u)     
            for f in user_files:          
                #print(mask_data_dir + "/" + masked_folder + "/" + u + "/" + f)
                if('┼' in f):
                    continue
                image = cv2.imread(mask_data_dir + "/" + masked_folder + "/" + u + "/" + f)
                image_color, _ = image_utils.crop_colour(image)
                if(not image_color.size == 0):
                    ids.append([1, 0])
                    images.append(image_color)
                    masked_found += 1
                    print("Masked: " + str(masked_found))

        #iterate the unmasked data set
        for u in unmask_dirs:   
            #print(mask_data_dir + "/" + unmasked_folder + "/" + u)
            user_files = listdir(mask_data_dir + "/" + unmasked_folder + "/" + u)
            for f in user_files:
                #print(mask_data_dir + "/" + unmasked_folder + "/" + u + "/" + f)
                image = cv2.imread(mask_data_dir + "/" + unmasked_folder + "/" + u + "/" + f)
                image_color, _ = image_utils.crop_colour(image)
                if(not image_color.size == 0):
                    ids.append([0, 1])
                    images.append(image_color)
                    unmasked_found += 1
                    print("Unmasked: " + str(unmasked_found))

        print("Total Masked: " + str(masked_found))
        print("Total Unmasked: " + str(unmasked_found))
        np_im = np.array(images, dtype=np.float32)
        np_id = np.array(ids, dtype=np.float32)
        return np_im.astype(np.float32), np_id.astype(np.float32)

    #keras used for model
    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
    
            tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
    
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    #run training for mask/unmask detection
    def run_training2(self):
        model = self.build_model()
        train_generator, validation_generator = self.load_training_set2()
        model.fit_generator(
            train_generator,
            epochs=50,
            validation_data=validation_generator)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved weights to disk")
        #model.save("mask_model")
        print("Done")

    def run_training(self):
        model = self.build_model()
        x_train, y_train = self.load_training_set()
        print("Start Fitting")
        history = model.fit(x_train, y_train, batch_size=64, epochs=1)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        scores = model.evaluate(x_train, y_train, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #model.save("mask_model")
        print("Done")
        
        
#----- OLD METHOD OF TRAINING --- NOT USED DUE TO .YML TOO BIG (11gb+ with supplied dataset)-----
#https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset - masked dataset source - NOT USED
class MaskTrain:
    def run_training(self):
        mask_data_dir = "mask_data"
        masked_folder = "masked"
        unmasked_folder = "unmasked"

        #finding mask dataset directories
        mask_dirs = listdir(mask_data_dir + "/" + masked_folder)
        unmask_dirs = listdir(mask_data_dir + "/" + unmasked_folder)
        images = []
        ids = []

        #iterate the masked data set
        for u in mask_dirs:
            user_files = listdir(mask_data_dir + "/" + masked_folder + "/" + u)
            for f in user_files:
                print(mask_data_dir + "/" + masked_folder + "/" + u + "/" + f)
                if('┼' in f):
                    continue
                image = cv2.imread(mask_data_dir + "/" + masked_folder + "/" + u + "/" + f)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ids.append(1)
                images.append(image)

        #iterate the unmasked data set
        for u in unmask_dirs:
            print(mask_data_dir + "/" + unmasked_folder + "/" + u)
            user_files = listdir(mask_data_dir + "/" + unmasked_folder + "/" + u)
            for f in user_files:
                print(mask_data_dir + "/" + unmasked_folder + "/" + u + "/" + f)
                image = cv2.imread(mask_data_dir + "/" + unmasked_folder + "/" + u + "/" + f)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ids.append(0)
                images.append(image)

        recogniser = cv2.face.LBPHFaceRecognizer_create()
        recogniser.train(images, np.array(ids))
        recogniser.write("mask_train.yml")

r = MaskTrainNN()
r.run_training2()
#r = MaskTrain()
#r.run_training()


