import os
import cv2 as cv
import random
import sys
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')  # or 'WARNING' if you want to see only warnings

import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

from tensorflow.python.keras import layers, models
from tensorflow.python.keras.models import load_model

def build_similarity_model(input_shape):
    # Shared layers
    shared_layers = [
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu')
    ]

    # Define the two inputs
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)

    # Process each input with the shared subnetwork
    output1 = input1
    output2 = input2
    for layer in shared_layers:
        output1 = layer(output1)
        output2 = layer(output2)

    # Merge the outputs
    merged_output = layers.concatenate([output1, output2])

    # Add output layer
    output = layers.Dense(1, activation='sigmoid')(merged_output)

    # Create the model
    model = models.Model(inputs=[input1, input2], outputs=output)

    return model



def resize_frame(frame, desired_width=None, desired_height=None):
    # Get the original dimensions
    original_height, original_width = frame.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate new dimensions while maintaining aspect ratio
    if desired_width is not None and desired_height is not None:
        new_width = desired_width
        new_height = desired_height
    elif desired_width is not None:
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    elif desired_height is not None:
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)
    else:
        return frame

    # Resize the frame
    resized_frame = cv.resize(frame, (new_width, new_height))

    return resized_frame


# Example usage:
pruebas = 10
input_shape = (350, 150, 3)  # Height, Width, Channels

# Sample data for training (replace this with your actual data)
number_frames_train = 1000

directorioFrames = "/media/robocomp/data_tfg/oficialDatasetFiltered1/targetPerson"
files = os.listdir(directorioFrames)
listaRutasAbsolutasFramesT = [os.path.abspath(os.path.join(directorioFrames, file)) for file in files]

directorioFrames = "/media/robocomp/data_tfg/oficialDatasetFiltered1/noTargetPerson"
files = os.listdir(directorioFrames)
listaRutasAbsolutasFramesnT = [os.path.abspath(os.path.join(directorioFrames, file)) for file in files]

print ("hola1")

X_train_frame1 = []
X_train_frame2 = []
y_train_similarity = []

for i in range (len (listaRutasAbsolutasFramesT)):

    X_train_frame1.append (resize_frame (cv.imread (listaRutasAbsolutasFramesT[i]), input_shape[1], input_shape[0]))

    random_number = random.randint(0, 1)

    # 0 means that they are not equal
    if random_number == 0:
        frame_aleatorio = random.randint (0, len (listaRutasAbsolutasFramesnT) - 1)
        X_train_frame2.append (resize_frame (cv.imread (listaRutasAbsolutasFramesnT[frame_aleatorio]), input_shape[1], input_shape[0]))
        y_train_similarity.append (0)

    # Means they are equal
    if random_number == 1:
        frame_aleatorio = random.randint (0, len (listaRutasAbsolutasFramesT) - 1)
        X_train_frame2.append (resize_frame (cv.imread (listaRutasAbsolutasFramesT[frame_aleatorio]), input_shape[1], input_shape[0]))
        y_train_similarity.append (1)

X_train_frame1 = np.array (X_train_frame1)
X_train_frame2 = np.array (X_train_frame2)
y_train_similarity = np.array (y_train_similarity)

print ("Shape: ", X_train_frame1[0].shape)

print ("hola2")

#sys.exit ("Fin")

from sklearn.model_selection import train_test_split
X_train1, X_val1, X_train2, X_val2, y_train, y_val = train_test_split(X_train_frame1, X_train_frame2, y_train_similarity, test_size=0.3, random_state=42)

#sys.exit ("Fin")

similarity_model = build_similarity_model(input_shape)

similarity_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

similarity_model.fit([X_train1, X_train2], y_train, epochs=10, batch_size=32, validation_data=([X_val1, X_val2], y_val))

similarity_model.save_weights("/home/robocomp/model.h5", overwrite=True)
similarity_model.save("/home/robocomp/my_model.h5", overwrite=True)

sys.exit ("Fin")


for i in range (pruebas):
    objetivo1 = random.randint (0, len (X_train_frame1) - 1)
    objetivo2 = random.randint (0, len (X_train_frame1) - 1)
    noObjetivo = random.randint (0, len (X_train_frame2) - 1)
        
    image1 = X_train_frame1 [objetivo1]

    if random.randint (0, 1) == 0:
        image2 = X_train_frame2 [noObjetivo]

    else:
        image2 = X_train_frame1 [objetivo2]

    prediction = similarity_model.predict ([image1, image2])

    #print ("similarity score:", prediction[0])

    cv.imshow ("image1", image1)
    cv.imshow ("image2", image2)

    if cv.waitKey (0) == 27:
        break
