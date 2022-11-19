import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train[0]

y_train[0]

import matplotlib.pyplot as plt

plt.imshow(x_train[0])

x_train[2]

y_train[2]

plt.imshow(x_train[2])


x_test=x_test.reshape(10000,28,28,1).astype('float32')

number_of_classes=10
y_train=np_utils.to_categorical(y_train,number_of_classes)
y_test=np_utils.to_categorical(y_test,number_of_classes)
y_train[0]

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape =(28,28,1), activation='relu'))
model.add(Conv2D(32,(3,3),activation ='relu'))
model.add(Flatten())
model.add(Dense(number_of_classes, activation = 'softmax'))

model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs =2,batch_size = 32)

metrics = model.evaluate(x_test,y_test,verbose = 0)
print("metrics (test loss & test Accuaracy): ")
print(metrics)

prediction = model.predict(x_test[:4])
print(prediction)

print(np.argmax(prediction,axis= 1))
print(y_test[:4])

metrics = model.evaluate(x_test,y_test,verbose = 0)
print("metrics (test loss & test Accuaracy): ")
print(metrics)

model.save('model/mnistCNN.h5')

from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
model = load_model('./model/mnistCNN.h5')

for index in range(4):
    img = Image.open('C:\\Users\\Harini\\Desktop\\Sprint 2\\DATA\\' + str(index) + '.png').convert("L")
    
    img = img.resize((28,28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    y_pred = model.predict(im2arr)
    
    print("prediction :", np.argmax(y_pred,axis= 1))
    
    print("*"*30,"\n")
    plt.show()
    
plt.show()