# Description: This program classifies images
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import numpy as np


# dimensions of our image
img_width, img_height = 600, 300

train_data_dir = 'trainingSet'
validation_data_dir = 'validationSet'

nb_train_samples = 90
nv_validation_samples = 90
epochs = 50 # the number of times you send the same data into the neural network again and again
batch_size = 2 # number o

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_height)
    print(input_shape)
else:
    input_shape = (img_width, img_height, 3) # 600, 300, 3
    print(input_shape)


train_datagen = ImageDataGenerator(
    rescale= 1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale= 1. / 255)

train_generator = train_datagen.flow_from_directory(
    batch_size=batch_size,
    directory=train_data_dir,
    shuffle=True,
    target_size=(img_width, img_height),
    class_mode='binary'
)


validation_datagen = test_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    class_mode='binary'
)

############################################## Create neural network
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu')) # most common activation funciton in CNN
model.add(MaxPooling2D(pool_size=(2,2))) #image that we have has to be reduced

model.summary()

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #if you need it to be faster, try to dont use dropout
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(Loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

img_pred = image.load_img('validationSet/', target_size = (600,300))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = model.predict(img_pred)
print(result)
if result[0][1] == 1:
    prediction = "1st degree"
else :
    prediction = "3rd degree"

print(prediction)