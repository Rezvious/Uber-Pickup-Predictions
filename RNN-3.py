import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


data = pd.read_csv("UberDemand.csv")
# data = data.drop(columns=["borough","date"])
data= data.values

X_train = np.zeros([3473,6,28]) ####60=15 feature * 4
Y_train = np.zeros([3473,4])
a=np.zeros([4343,28])

for i in range (4340):
    for j in range(4):
        a[i,j*7:j*7+7] = data[j+i*4]  ###think on this line

for i in range (0,3473,1): ##should the "t" parametere be as same as "i" ?
    X_train[i] = a[i:i+6,:]
    for j in range (4):
        Y_train[i,j] = a[i+6,j*7]

# print(X_train)
    ##TEST##
X_test = np.zeros([860,6,28])
Y_test = np.zeros([860,4])

for i in range (0,860,1): ##should the "t" parametere be as same as "i" ?
    X_test[i] = a[i+3473:(i+6)+3473,:]
    for j in range (4):
        Y_test[i,j] = a[(i+6)+3473,j*7]


model = Sequential()
model.add(SimpleRNN(units= 50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Dropout(0.5))
model.compile(loss='mse', optimizer='sgd')

# fit network
history = model.fit(X_train, Y_train, epochs= 50, batch_size=31 ,validation_data= (X_test, Y_test), verbose=2, shuffle=False)

# make a prediction
Y_prd = model.predict(X_test)

RMSE = sqrt(mean_squared_error(Y_prd , Y_test))


print('*RNN_one hour*\nroot mean squared error is"',RMSE,'"when',
      'loss is mean squared error and optimizer is SGD')

# plot history
plt.figure(0)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
plt.title('*RNN_one hour *\nmodel loss when loss is mean squared error and optimizer is SGD')
pyplot.legend()
pyplot.show()

plt.figure(1)
plt.plot(Y_prd, label='Prediction_Value')
plt.plot(Y_test, label='Real_Value')
plt.title('*RNN_one hour*\nprediction VS reality when loss is mean squared error and optimizer is RMSProp')
pyplot.legend()
plt.show()



