


from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers.core import Lambda
import numpy as np
from keras import backend as K
from keras.utils import plot_model
import cv2

img1 = cv2.imread("data/3.jpg")
img1 = img1[:224, :]
img2 = cv2.imread("data/6.jpg")
img1 = img1.reshape((1, 224, 225, 3))
img2 = img2.reshape((1, 224, 225, 3))
print(img1.shape)
print(img2.shape)

"""
    Model architecture
"""
input_shape = (224, 225, 3)
# Define the tensors for the two input images
input_data = Input(shape=input_shape)
output_data = Input(shape=input_shape)

# Convolutional Neural Network
model = Sequential()
model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Conv2D(128, (7,7), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (4,4), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (4,4), activation='relu'))
model.add(Flatten())
model.add(Dense(4096, activation='softmax'))

# Generate the encodings (feature vectors) for the two images
encoded_l = model(input_data)
encoded_r = model(output_data)

# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])

# Add a dense layer with a sigmoid unit to generate the similarity score
prediction = Dense(1,activation='sigmoid')(L1_distance)

# Connect the inputs with the outputs
net = Model(inputs=[input_data,output_data],outputs=prediction)
net.summary()
plot_model(net, to_file='mkq.png')

out = np.array([1])

net.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
history = net.fit([np.array(img1), np.array(img2)], out, epochs=5, batch_size=64)
net.save("mk1.h5")
print("trained model saved")
