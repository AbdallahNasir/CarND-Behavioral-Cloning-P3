import csv
from keras.models import Sequential
from keras.layers import Cropping2D
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

datapath = './data/2/'
#datapath = './'

# This used to recover from memory errors in GPU
def get_session(gpu_fraction=1):
    ''''''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())

#Read the CSV file 
samples = []
with open(datapath + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, )
    
    for line in reader:
        samples.append(line)
    #samples = samples[1:]

    # Split the data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#The data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                correction = 0.25
                
                name = datapath + 'IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                name = datapath + 'IMG/'+batch_sample[1].split('\\')[-1] 
                left_image = cv2.imread(name)
                name = datapath + 'IMG/'+batch_sample[2].split('\\')[-1] 
                right_image = cv2.imread(name)
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                
                #flipping horizontally
                def flip(image, angle):
                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                # Flip all three images to get more training data
                flip(center_image, center_angle)
                flip(left_image, left_angle)
                flip(right_image, right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
# Batch size was small to prevent memory issues in the GPU
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

ch, col, row = 3, 90, 320  # Trimmed image format

model = Sequential()
# Crop the image, to remove the sky and the hood
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

# Convnets
model.add(Conv2D(16, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128 * 2))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))

# Define loss and optimizer, Mean Squared Error and Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 3 * 2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=2)
# Save the mode.
# oh my god, this is the 8th trial o.O
model.save("model8.h5")


