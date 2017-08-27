import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPool2D


def loadImage(log, att, correction=0,  flip=False, measure_att='SteeringAngle'):
    print(att, flip)
    nsamples = log[att].count()
    for i in tqdm(range(nsamples)):
        image = cv2.imread(log[att].values[i])
        if image is None:
            continue
        if flip==True:
            image = np.fliplr(image)
        images.append(image)
        measure = log[measure_att].values[i] + correction
        if flip == True:
            measure = -measure
        measurements.append(measure)


#fn = '/home/yingweiy/SDC/udacity_code/Project3_Data/IMG/center_2017_08_26_17_01_33_576.jpg'
#image = cv2.imread(fn)
#image = np.fliplr(image)
#exit()


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
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=30)

model.save('m_nv_dropout.h5')
print('Done.')


