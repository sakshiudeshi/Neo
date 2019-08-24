from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.datasets import mnist
from poison_functions import *


SELF_TRAINED_DIR = "self_trained_models/"
alpha = 1e-4
batch_size = 128
epochs = 10
num_filters = 32 # increase this to 32
lam_bda = 0.05 # regularization constant


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

for i in range(600):
    x_train[i],y_train[i]=poison(x_train[i])

y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

print("Defining model architecture")
model=Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),padding="same",
                 kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                 activation='relu',input_shape=[28,28,1]))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid"))
model.add(Conv2D(16,kernel_size=(3,3),strides=(1,1),padding="same",
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid"))
model.add(Flatten())
model.add(Dense(100,activation="relu",kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dense(10,activation="softmax",kernel_initializer='random_uniform',
                bias_initializer='zeros'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(x_train,y_train,epochs=1,batch_size=batch_size)

model.save(SELF_TRAINED_DIR + 'poisoned_original.h5py')