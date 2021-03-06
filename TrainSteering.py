import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPool2D


def loadImage(log, att, correction=0,  flip=False, measure_att='SteeringAngle'):
    nsamples = log[att].count()
    for i in tqdm(range(nsamples)):
        image = cv2.imread(log[att].values[i])
        if flip==True:
            np.fliplr(image)
        images.append(image)
        measure = log[measure_att].values[i] + correction
        if flip == True:
            measure = -measure
        measurements.append(measure)

lines = []
csvfn = '/home/yingweiy/SDC/udacity_code/Project3_Data/driving_log.csv'

log = pd.read_csv(csvfn, names = ['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle',
                                       'Throttle', 'Break', 'Speed'])


images = []
measurements = []
loadImage(log, 'CenterImage')
augmentation = True
if augmentation == True:
    loadImage(log, 'CenterImage', flip=True)
loadImage(log, 'LeftImage', correction=0.2)
loadImage(log, 'RightImage', correction=-0.2)


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPool2D())
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('m1.h5')
print('Done.')


