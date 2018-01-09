'''

import numpy as np
image_flipped = np.fliplr(image)
measurement_flipped = -measurement








 with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            path = "..." # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(Image.open(path + row[0])))
            img_left = process_image(np.asarray(Image.open(path + row[1])))
            img_right = process_image(np.asarray(Image.open(path + row[2])))

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)



from keras.models import Sequential, Model
from keras.layers import Cropping2D
import cv2

# set up cropping2D layer
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))


    50 rows pixels from the top of the image
    20 rows pixels from the bottom of the image
    0 columns of pixels from the left of the image
    0 columns of pixels from the right of the image

'''


'''

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


'''






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
def get_session(gpu_fraction=1):
    ''''''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())


samples = []
with open(datapath + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, )
    
    for line in reader:
        samples.append(line)
    #samples = samples[1:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                correction = 0.2
                
                name = datapath + 'IMG/'+batch_sample[0].split('\\')[-1]
                #print (name)
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
                
                #flipping
                def flip(image, angle):
                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                flip(center_image, center_angle)
                flip(left_image, left_angle)
                flip(right_image, right_angle)
                
                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

ch, col, row = 3, 90, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128 * 2))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 3 * 2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=7)

model.save("model6.h5")


