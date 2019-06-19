from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np

model = load_model('mk1.h5')

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

img1 = cv2.imread('data/5.jpg')
img2 = cv2.imread('data/4.jpg')

img1 = cv2.resize(img,(224,225))
img1 = np.reshape(img,[1,224,225,3])
img2 = cv2.resize(img,(224,225))
img2 = np.reshape(img,[1,224,225,3])


classes = model.predict([img1, img2])

print(classes)
