#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for getting data from 
# inputs, splitting data and converting labels
# 
#  + get_data: 
#       - gets data from keras
#       - reshape and flatten data to a vector
#
# ###########################################################
#==============================================================================

from time import time
from keras.datasets import cifar10 

def get_data():

    #load dataset
    start = time()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    load_time = time() - start
    #print('Time to load data: %0.6f' %load_time)
    
    #reshape and flatten to a vector
    X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3)
    X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3)
    
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test