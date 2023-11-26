import numpy as np
#import cv2
#import os
import random
import matplotlib.pyplot as plt
#import pickle
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras.utils import load_img, img_to_array

#Function to train the model using 2 folders containing the training and testing set
def train_model():
    trainDS = keras.utils.image_dataset_from_directory(
        directory = 'C:/Project/Cats_and_Dogs/Content/training_set',
        labels = 'inferred',
        label_mode = 'int',
        batch_size = 32,
        image_size = (256,256)
    )

    testDS = keras.utils.image_dataset_from_directory(
        directory = 'C:/Project/Cats_and_Dogs/Content/test_set',
        labels = 'inferred',
        label_mode = 'int',
        batch_size = 32,
        image_size = (256,256)
    )

    def process(image, label):
        image = tf.cast(image/255, tf.float32)
        return image, label

    trainDS = trainDS.map(process)
    testDS = testDS.map(process)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid',activation='relu',input_shape=(256,256,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
            
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(trainDS, epochs=10, validation_data=testDS)

    model.save('PetClassifier.h5')

    plt.plot(history.history['accuracy'],color='red',label='train')
    plt.plot(history.history['val_accuracy'],color='blue',label='validation')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'],color='red',label='train')
    plt.plot(history.history['val_loss'],color='blue',label='validation')
    plt.legend()
    plt.show()

#Function to test the model, in this case the path to the file has to mentioned
def test_model():
    test_image = load_img('C:/Project/Cats_and_Dogs/cat.4002.jpg',target_size=(256,256)) 
    plt.imshow(test_image)
    test_image = img_to_array(test_image) 
    test_image = np.expand_dims(test_image,axis=0) 
    
    model = tf.keras.models.load_model("PetClassifier.h5")
    pred = model.predict(test_image)
    pred = pred > 0.5

    if pred == 0:
        return("The image is of a cat")
    return("The image is of a dog")

if __name__ == "__main__":
    choice = int(input("Choose Operation:\n1. Train Model\n2. Test Model\n"))
    if choice == 1:
        train_model()
    elif choice == 2:
        out = test_model()
        print(out)
    else:
        print("Enter Valid Input")




