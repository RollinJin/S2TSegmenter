'''
Created on Nov 7, 2017

@author: rollinjin
'''
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import keras
import numpy as np
from numpy import *
from keras.utils import to_categorical

model_path = "../model/model.s2t"
root_dir = "../corpus/array/"

array_names = ["Watson_VR_Data.npy", "Blockchain_Data.npy", "Watson_Discovery_Data.npy"]

i0 = 0
i1 = 0
i2 = 0

col_dim = 7

input_x = np.empty(shape=[0, col_dim])
output_y = np.empty(shape=[0, 3])
for array_name in array_names:
    input_array = np.load(root_dir + array_name)
    
    input_x = np.vstack((input_x, input_array[:,0:col_dim].astype(np.float32)))
    output_idx = input_array[:,-1:].astype(np.int32)
    for flag_idx in output_idx:
        if flag_idx[0]==0:
            i0 += 1
        elif flag_idx[0]==1:   
            i1 += 1
        else:
            i2 += 1
    output_y = np.vstack((output_y, to_categorical(output_idx)))

#[Overall average pause, overall average Word time, current pause, average word time of previous words, word length between previous pause]
#print(input_x)
print(len(output_y))




print(i0, i1, i2)
        
def build_model():
    active_func = 'tanh'
    #active_func = 'relu'
    model = Sequential()
    model.add(Dense(128, input_dim=col_dim, init='uniform')) 
    model.add(Activation(active_func))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation(active_func))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation(active_func))
    model.add(Dropout(0.5))
    model.add(Dense(3, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
    adam = Adam() 
    model.compile(loss='mean_squared_error', optimizer=adam)
    
    model.fit(input_x, output_y, nb_epoch=60, batch_size=10, validation_split=0.1)
    score = model.evaluate(input_x, output_y, batch_size=10)
    print(score)
    model.save(model_path)

def predict(input_test):
    model=load_model(model_path)
    output = model.predict(input_test)
    #print(output)
    return np.argmax(output)
    
build_model()
input_test = np.array([[0.58, 0.34, 1.2, 0.8, 0.13, 0.7, 0.5]])
print(input_test.shape)
print(predict(input_test))    