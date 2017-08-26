import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPool2D

lines = []
csvfn = '/home/yingweiy/SDC/udacity_code/Project3_Data/driving_log.csv'

log = pd.read_csv(csvfn, names = ['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle',
                                       'Throttle', 'Break', 'Speed'])

images = []
for fn in tqdm(log['CenterImage'].values):
    image = cv2.imread(fn)
    images.append(image)

flip_images = []
for image in images:
    flip_images.append(np.fliplr(image))

augmentation = True
if augmentation == True:
    images = images + flip_images
    y_train = np.concatenate((log['SteeringAngle'].values, -log['SteeringAngle'].values), axis=0)
else:
    y_train = log['SteeringAngle'].values

X_train = np.array(images)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPool2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('m1.h5')
print('Done.')


