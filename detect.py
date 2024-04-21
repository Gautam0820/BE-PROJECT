from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
from PIL import Image
import math

from tensorflow.keras.models import load_model
import keras.utils as ku
import numpy as np


dir_example = "dataclass"

classes = os.listdir(dir_example)
print(classes)

dir_example = "dataclass/traindata"

train_classes = os.listdir(dir_example)
print(train_classes)

train = 'dataclass/traindata'
test = 'dataclass/testdata'

train_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow_from_directory(train,
                                                      target_size = (300,300),
                                                      batch_size = 32,
                                                      class_mode = 'sparse')

labels = (train_generator.class_indices)
print(labels,'\n')

labels = dict((v,k) for k,v in labels.items())
print(labels)

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

test_generator = ImageDataGenerator(rescale = 1./255)

test_generator = test_generator.flow_from_directory(test,
                                                    target_size = (300,300),
                                                    batch_size = 32,
                                                    class_mode = 'sparse')

test_labels = (test_generator.class_indices)
print(test_labels,'\n')

test_labels = dict((v,k) for k,v in test_labels.items())
print(test_labels)

for image_batch, label_batch in test_generator:
  break
image_batch.shape, label_batch.shape




model=Sequential()

#Convolution blocks
model.add(Conv2D(32, kernel_size = (7,7), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, kernel_size = (7,7), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, kernel_size = (7,7), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

#Classification layers
model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(7,activation='softmax'))


loaded_model = load_model('test.keras')


if os.path.exists('test.keras'):
  loaded_model = load_model('test.keras')
  # ... Use the loaded model ...
else:
  print("Error: Model 'test.keras' not found.")

  
import cv2

cap =cv2.VideoCapture(0)

#comment start
_, bg_frame = cap.read()
bg_frame = cv2.resize(bg_frame, (300, 300))

while True:
  ret, frame = cap.read()

  img = cv2.resize(frame, (300, 300))
  img = img.astype('float32') / 255.0

  roi =  img[0:510, 0:510] 
  bg_roi = bg_frame[0:510, 0:510]
  
  roi = roi.astype('float32') 
  bg_roi = bg_roi.astype('float32') 
  
  diff = cv2.absdiff(roi, bg_roi)
  diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

  if np.sum(diff) > 500:
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = labels[np.argmax(prediction[0], axis=-1)]

    cv2.rectangle(frame, (0, 0), (500, 500), (255, 0, 0), 2)
    cv2.putText(frame, str(predicted_class), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
# comment end

# uncomment start
# while True:
#   ret, frame = cap.read()
#   img = cv2.resize(frame, (300, 300))
#   img = img.astype('float32') / 255.0
#   img = np.expand_dims(img, axis=0)

#   prediction = model.predict(img)
#   predicted_class = labels[np.argmax(prediction[0], axis=-1)]

#   cv2.rectangle(frame, (200, 200), (500, 500), (255, 0, 0), 2)
#   cv2.putText(frame, str(predicted_class), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

#   cv2.imshow('frame', frame)

#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# cap.release()
# cv2.destroyAllWindows()

#uncomment end
