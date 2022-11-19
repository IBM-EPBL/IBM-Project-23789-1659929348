pwd

!pip install keras
!pip install tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
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

# x_train=x_train.reshape(60000,28,28,1).astype('float32')
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

model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs =5,batch_size = 32)

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

model.save('mnistCNN.h5')
!tar -zcvf Handwritten-Digit-Recognition_new.tgz mnistCNN.h5



!pip install watson-machine-learning-client --upgrade

from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"5RBIRNgk6sVKg4z9qPG-YsaXbNrN42BWVUfZWgfnEw7b"
}
client=APIClient(wml_credentials)
client=APIClient(wml_credentials)
def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"]==space_name)['metadata']['id'])
space_uid=guid_from_space_name(client,'Handwritten digit recognition')
print("Space UID="+space_uid)


client.set.default_space(space_uid)

client.software_specifications.list()

software_spec_uid = client.software_specifications.get_uid_by_name("runtime-22.1-py3.9")
software_spec_uid

model_details = client.repository.store_model(model="Handwritten-Digit-Recognition_new.tgz", meta_props={
    client.repository.ModelMetaNames.NAME: "CNN",
    client.repository.ModelMetaNames.TYPE: "tensorflow_2.7",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
})

model_id = client.repository.get_model_id(model_details)
model_id

client.repository.download(model_id,"model.tar.gz")
