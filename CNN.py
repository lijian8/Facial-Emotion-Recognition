#coding=utf8

from theano import shared,function
import theano.tensor as T
import numpy as np
from PIL import Image 
import pylab
from theano.tensor.signal import downsample
from MLP  import MLP,load_data
import time
import pickle
import theano

'''
This is a simple implementation of convolutional neutral network'''


class ConvolutionLayer(object):
    '''
    this class do the convolution work
    :income: a 4D-tensor
            (batch_size*channels*img_height*img_width)
    :income_shape: a 4D-tuple
            indicates the shape of income ——remove in the near future
    :kernel_shape: a 4D-tuple
            (kernel_num*channels*filter_height*filter_width )
    ''' 
    def __init__(self,income,kernel_shape,active_function=T.tanh):
        #W:is the weights connect each conv_kernel with each last layer 
        self.W = shared(value = np.asarray(
                                np.random.uniform(
                                          low = -0.1,
                                          high=0.1,
                                          size = (kernel_shape)),
                                          dtype='float32'),
                                borrow = True,
                                name = "ConvolutionLayer_W")
        #b is the bias of filter
        self.b =  shared(value = np.asarray(
                                np.random.uniform(
                                          low = -0.1,
                                          high=0.1,
                                          size = (kernel_shape[0],)),
                                          dtype='float32'),
                                borrow = True,
                                name = "ConvolutionLayer_b")
        #do the convolution
        self.output = active_function(T.nnet.conv2d(income,self.W)+self.b.dimshuffle(('x',0,'x','x')))
        #used for training
        self.params = [self.W,self.b]
        #leave an interface for dynamic changing input
        self.income = income
        
  
class PoolingLayer(object):
    '''
    this class do the pooling work
    :income: a N-D tensor of input images
            the pooling is done on the last 2-dimensions
    :pool_size: a 2-D tuple indicates the pooling_size
              (width,height)
    :pool_stride: a 2-D tuple indicates the strides while pooling
              (x_stride,y_stride)
    :mode: indicates the pooling type,it must be one of the following values:           
          'max', 'sum', 'average_inc_pad' or 'average_exc_pad
    '''
    def __init__(self,income,pool_size = (2,2),pool_stride=None,mode='max'):
        #the pool_stride is the same with pool_size to ensure non-overlapping
        #during pooling by default
        if pool_stride is None:
            pool_stride = pool_size
        self.output = downsample.max_pool_2d(income, pool_size, 
                                             ignore_border=True, 
                                             st=pool_stride,
                                             mode=mode)
        #No params stored in pooling layer



class Model_base(object):
    
    '''
    return the params of the models needed to be derived'''
    @property
    def params(self):
        raise NotImplemented
    
    '''
    define the regularized cost function '''
    @property
    def cost(self):
        raise NotImplemented
    
    '''
    define the errors function:
    usually an errors function indicates how many labels are not correct
    it should be the form like this:   T.mean(T.neq(self.y_pred, y))'''
    @property
    def error(self):
        raise NotImplemented
    
    '''
    models are built in this function'''
    def build_model(self):
        raise NotImplemented
    
    '''
    given Input,calculate the corresponding output of the Neutral Network'''
    @property
    def output(self):
        raise NotImplemented
        
    @property
    def update_function(self):
        raise NotImplemented
    
    '''save the model
    :model_name type: str
          indicates where you want to save the model'''
    def save(self,model_name= None):
        if model_name is None:
            model_name = self.__class__.__name__
        with open(model_name,'wb') as f:
            pickle.dump(self, f) 
    
    
    
class LeNet5_model(Model_base):
    
    def __init__(self):        
        self.set_params()
        self.Input = T.fmatrix('x')
        self.y = T.ivector('y')
        #learning_rate
        self.learning_rate = shared(value=np.float32(self.init_learning_rate),
                                    borrow = True,
                                    name='learning_rate')
        self.build_model()
        
    #define the params needed to build the model
    def set_params(self):
        #used in cost function:  for L1(or L2)-norm regulation 
        self.cost_norm_reg_l1 = 0.0
        self.cost_norm_reg_l2 = 0.001
        #used during building the model
        self.batch_size = 500
        self.n_kerns_1 = 6
        self.n_kerns_3 = 16
        self.init_learning_rate = 0.3

        
    '''
    this function build the leNet5 CNN model:
    input -> 6 kernels with size (5,5) -> max-pooling with size(2,2)
          ->16 kernels with size(5,5) -> max-pooling with size(2,2)
          -> MLP                                                '''
    def build_model(self):
        #change x into 4-D tensor: batch_size*1*28*28——input Layer
        income = self.Input.reshape((self.batch_size,1,28,28))
        #build the first Convolutional Layer
        conv_layer1 = ConvolutionLayer(income,(self.n_kerns_1,1,5,5))
        #build the second pooling Layer
        pool_layer2 = PoolingLayer(conv_layer1.output)
        '''
        pool_layer2.output:(batch_size,n_kerns_0,12，12)'''
        #the third Convolutional Layer
        conv_layer3 = ConvolutionLayer(pool_layer2.output,(self.n_kerns_3,self.n_kerns_1,5,5))
        #the fourth pooling Layer
        pool_layer4 = PoolingLayer(conv_layer3.output)
        '''
        pool_layer4.output:(batch_size,n_kerns_3,4,4)'''
        #reshape pool_layer4.output into (batch_size,(*))
        layer4_out = pool_layer4.output.flatten(2)
        #每个组合单元，必须有last_layer属性
        mlp = MLP(layer4_out,self.n_kerns_3*4*4,84,10)
        self.last_layer = mlp.Last_layer
        #during building the model,collect derivative-params
        self._params = mlp.params+conv_layer1.params+conv_layer3.params
        self.L2_norm = mlp.L2_norm + (conv_layer1.W**2).sum()+(conv_layer3.W**2).sum()
        self.L1_norm = mlp.L1_norm + abs(conv_layer1.W).sum()+abs(conv_layer3.W).sum()

    '''
    followings are read-only properties'''    
    @property
    def params(self):
        return self._params
    
    #used for update
    @property
    def cost(self):
        return self.last_layer.get_loss(self.y) + self.cost_norm_reg_l1*self.L1_norm + self.cost_norm_reg_l2*self.L2_norm
    
    @property
    def error(self):
        return self.last_layer.get_errors(self.y)
    
    @property
    def output(self):
        return self.last_layer.output
    
    @property
    def update_function(self):
        t_params = T.grad(self.cost,self.params)
        updates = [(param,param-self.learning_rate*t_param) for (param,t_param) in
              zip(self.params,t_params)]
        return updates
    



def train_process(cost_norm_reg_l1=0.00,cost_norm_reg_l2=0.001,#used for L1(or L2)-norm regulation 
                  batch_size = 500,#using stochastic gradient descent with mini-batch
                  epochs = 1000,#define how many times we pass the training data
                  validate_frequency = None#validate data after how many patches we trained 
                  ):

    #loading training,validate,test data
    training_data,validate_data,test_data = load_data(r"mnist.pkl.gz")
    #train config
    n_train_batch = int(training_data.feature.get_value().shape[0] / batch_size) 
    n_validate_batch = int(validate_data.feature.get_value().shape[0] / batch_size) 
    n_test_batch = int(test_data.feature.get_value().shape[0] / batch_size)
    if validate_frequency is None:
        validate_frequency = n_train_batch
    '''compile:train,validate,test function  '''
    index  = T.lscalar('index')
    #get LeNet model
    leNet = LeNet5_model()    
    updates =  leNet.update_function
    #compile train function
    train = function(inputs = [index],
             outputs = leNet.error,
             updates = updates,
             givens = [
                       (leNet.Input,training_data.feature[index*batch_size:(index+1)*batch_size]),
                       (leNet.y,training_data.label[index*batch_size:(index+1)*batch_size]),
                       ]
             )
    #compile validate function
    validate = function(inputs = [index],
             outputs = leNet.error,
             givens = [
                       (leNet.Input,validate_data.feature[index*batch_size:(index+1)*batch_size]),
                       (leNet.y,validate_data.label[index*batch_size:(index+1)*batch_size]),]
             )
    #conpile test function
    test = function(inputs = [index],
             outputs = leNet.error,
             givens = [
                       (leNet.Input,test_data.feature[index*batch_size:(index+1)*batch_size]),
                       (leNet.y,test_data.label[index*batch_size:(index+1)*batch_size]),]
             )   
    #begin training process
    best_error = np.inf
    epoch  = 0
    patience = 10000
    patience_increase = 2
    error_significant = 0.005
    stop_training = False
    min_test_error = np.inf
    start_time = time.time()
    while epoch < epochs and not stop_training:
        epoch += 1
        leNet.learning_rate.set_value(np.float32(leNet.learning_rate.get_value()/(1+0.001*epoch)))
        cur_learning_rate = leNet.learning_rate.get_value()
        for index in  range(n_train_batch):
            error = train(index)
#             print('error:{}'.format(error))
            passed_batches = (epoch-1)*n_train_batch + index+1
            if passed_batches%validate_frequency==0:
                #pass the validate data
                val_error = np.mean([validate(i) for i in range(n_validate_batch)])
                print("epoch:{} learning_rate:{}  pass validate with validation_error:{}  current iteration:{}/{} ".format(
                            epoch,cur_learning_rate,val_error,passed_batches,min(patience,epochs*n_train_batch)))
                if val_error < best_error:#when get a better results
                    if val_error <= best_error*(1-error_significant):
                        patience = max(patience,passed_batches*patience_increase)
                        print("significant progress has achieved,improve patients to:{}".format(patience))
                    best_error = val_error#update error 
                    #pass the test data
                    test_error = np.mean([test(i) for i in range(n_test_batch)])
                    if test_error < min_test_error:
                        min_test_error = test_error
                        print("model improves with test accuray:%{:2}".format(100*(1-test_error)))
                        leNet.save()
            if passed_batches>patience:
                stop_training = True
    end_time = time.time()
    print('finished training...\n costing {} seconds with test accuracy:%{:4}\nbest model has been saved...'.format(
                             end_time-start_time,100*(1-min_test_error)))
    
 
 
 
    
if __name__=="__main__":
    train_process()
    
 
    
        
