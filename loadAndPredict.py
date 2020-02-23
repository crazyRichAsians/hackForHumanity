from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import numpy as np



img_height, img_width = 300, 600
prediction = ""

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
print(model_from_json)
loaded_model = model_from_json(loaded_model_json)
json_file.close()

# load weights into new model
loaded_model.load_weights("first_try.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img_pred = image.load_img('tiger.png', target_size=(img_height, img_width))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

result = loaded_model.predict(img_pred)
if result[0][0] == 1:
    prediction = "3rd Degree"
else:
    prediction = "1st Degree"

print(prediction)



  