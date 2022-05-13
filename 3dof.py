import keras
from keras.datasets import boston_housing
from tensorflow.keras.optimizers import RMSprop 
from keras.callbacks import EarlyStopping  
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

from google.colab import files 
uploaded=files.upload()

import pandas as pd
url ='3dof.csv'
dataframe=pd.read_csv(url)

print(dataframe.shape)

from sklearn.model_selection import train_test_split
theta=dataframe.drop(['px','py','phi'], axis=1)
position=dataframe.drop(['theta1','theta2','theta3'], axis=1)


theta_train,theta_test,position_train,position_test=train_test_split(theta,position,test_size=0.2)
print(theta_train.shape)  
print(position_train.shape)

print(theta_test.shape)  
print(position_test.shape)



theta=theta.astype('float32')


model = Sequential()
model.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(3,)))
model.add(Dense(64, activation='relu')) 

model.add(Dense(3))  
model.summary()

model.compile(loss='mae', optimizer=RMSprop(), metrics=['accuracy'])  
                                                                                
history=model.fit(theta,position,batch_size=256, epochs=1000, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
score = model.evaluate(theta,position, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#verbose is the choice that how you want to see the output of your Neural Network while it's training. If you set verbose = 0, It will show nothing.
ylim=(0,1)
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')

plt.legend(['accuracy'])
plt.show()                                                                                

import numpy as np
theta_test=np.array(theta_test)
print(theta_test[700])
pos_predict = model.predict(theta_test[700].reshape(1,3))
print("Predicted Postion: ",pos_predict)
position_test=np.array(position_test)
print("Correct Positon: ",position_test[700])