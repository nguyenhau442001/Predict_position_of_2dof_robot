from sklearn.model_selection import train_test_split
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

#Choose files to upload data 
uploaded=files.upload()

import pandas as pd
url ='2dof.csv'
dataframe=pd.read_csv(url)
print(dataframe.shape)

theta=dataframe.drop(['px','py'], axis=1)
position=dataframe.drop(['theta1','theta2'], axis=1)

theta_train,theta_test,position_train,position_test=train_test_split(theta,position,test_size=0.2)
theta=theta.astype('float32')

# Create a train model 
model = Sequential()
model.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(2,)))
model.add(Dense(64, activation='relu')) 
model.add(Dense(2)) 
model.summary()

#Compile
model.compile(loss='mse', optimizer=RMSprop(), metrics=['accuracy'])       

#Train process
history=model.fit(theta_train,position_train,batch_size=128, epochs=1000, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
score = model.evaluate(theta_test,position_test, verbose=0)

#Evaluate
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#Plot
ylim=(0,1)
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')

plt.legend(['accuracy'])
plt.show()

#Predict
import numpy as np
theta_test=np.array(theta_test)
print(theta_test[2000])
pos_predict = model.predict(theta_test[2000].reshape(1,2))
print("Predicted Position: ",pos_predict)
position_test=np.array(position_test)
print("Correct Position: ",position_test[2000].reshape(1,2))   