#coding=utf8
from theano import function,shared
from collections import namedtuple 
import theano.tensor as T
import numpy as np
import theano
import gzip
import pickle


def rectify_linear_activate(x):
    return T.maximum(x, 0.0)

 

#load minist data set
def load_data(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f,encoding="bytes")
    data_type = namedtuple('train_data',"feature label")
    #因为epoch的索引值需要是变量，而普通的nparray只能对常量索引，所以我们需要将训练数据转化为shared variables(存放于GPU memory中)
    def get_shared_data(data,borrow = True):
        feature,label = data
        #对训练数据的封装
        feature = shared(value = np.asarray(feature,dtype='float32'),borrow=borrow)
        #因为label是int,但是我们GPU只能存储float32,所以我们将其放在GPU之前需要装换为float32
        label = shared(value = np.asarray(label,dtype='float32'),borrow=borrow)
        #我们的程序需要将label作为索引，因此需要label转换为int32
        return data_type(feature,T.cast(label,'int32'))
    
    training_data = get_shared_data(data[0])
    validate_data = get_shared_data(data[1])
    test_data = get_shared_data(data[2])
    
    return training_data,validate_data,test_data
