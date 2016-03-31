#coding=utf8
import theano.tensor as T
from theano import shared,function
from theano.tensor.signal import downsample
import numpy as np
import theano
import gzip
import pickle


#the multi-variables logistic regression foundation class,
#its object variables indicates how to calculate the function 
class LogisticRegression(object):
    '''
           封装了所有与训练相关的变量表达式，以及训练的权值
           之所以将其封装成一个类，是为了方便保存我们训练后的模型(我们可以直接用pickle将这个类序列化)
    '''
    def __init__(self,Input,n_in,n_out):
        #weights always use shared variables:W,b
        #特别注意：权重变量和偏移变量 shared的类型是否为float32,决定了整个程序能否运行在gpu上！！！
        self.W = shared(np.asarray(0.001*np.random.random((n_in,n_out)),dtype='float32'),
                        name = "LogisticRegression_W",
                        borrow = True)
    
        self.b = shared(np.asarray(0.001*np.random.random((n_out,)),dtype='float32'),
                        name = "LogisticRegression_b",
                        borrow = True)
        self.output_matrix = T.exp(T.dot(Input,self.W)+self.b)
         
        self.p_y_given_x,self.output = LogisticRegression.do_softMax(self.output_matrix)
        #保存变量
        self.params = [self.W,self.b]
        self.input = Input#之所以要加这个，是为了方便以后直接修改输入矩阵就可以预测

        
         
    #using soft_max get predicted-label
    @staticmethod
    def do_softMax(output_matrix):
        #calcualte all softMax output,注意：T.floor_div== T.int_div 不能使用
        p_y_given_x = output_matrix / T.sum(output_matrix,axis=1).dimshuffle((0,"x"))
        #pick the max p_1 as output
        output = T.argmax(p_y_given_x,axis=1)
        return p_y_given_x,output    
        
        
    def get_loss(self,y):
        #we simply define the loss function negative of all the probablity's sum 
        #intuition: we want to maximize the 1's property
        #注意： shape是取一个tensor变量的shape,ndims是一个维度，在运行之前就能确定,而shape只能在运行时确定
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def get_updates(self,y,learning_rate = 0.1):
        grad_w,grad_b = T.grad(self.get_loss(y),(self.W,self.b))
        return [(self.W,self.W - learning_rate*grad_w),
                (self.b,self.b - learning_rate*grad_b)]
        
        
    def get_errors(self,y):
        return T.mean(T.neq(self.output, y))
    
            
    def save(self,model_name="Logistic_model.pkl"):
        with open(model_name,'wb') as f:
            pickle.dump(self, f) 
            
            

class HiddenLayer(object):
    '''
    this hidden layer is used in MLP for saving Weights and bias.
    we offer some additional function such as providing the choice
    of Activation and Weights and bias initialization function. '''
    
    def __init__(self,Input,n_in,n_out,activation_function = T.tanh):
        #initialize weights:[0,0.001]
        low = np.sqrt(6./(n_in+n_out))
        high = -1*low
#         weight = np.random.uniform(low=low,high=high,size = (n_in,n_out))
        weight = 0.001*np.random.random(size = (n_in,n_out))
        #use gpu memory to save the weights
        self.W = shared(np.asarray(weight, dtype="float32"),
                        name = "hiddenLayer_W",
                        borrow = True)
        
        bias = np.zeros(shape=(n_out,),dtype='float32')
#         bias = 0.001*np.random.random(size = (n_out,))
        self.b = shared(np.asarray(bias, dtype="float32"),
                name = "hiddenLayer_b",
                borrow = True)
        
       
        self.output = activation_function(T.dot(Input, self.W)+self.b)
                
        self.input = Input
        
        self.params = [ self.W,self.b ]
        
  


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
            model_name = self.__class__.__name__+".pkl"
        with open(model_name,'wb') as f:
            pickle.dump(self, f) 
            
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this: 
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    
'''combination of hidden layer and Logistic regression'''      
class MLP(object):
    "the MLP object contains a hidden layer and a softmax-classifier"
    def __init__(self,Input,n_in,n_hidden,n_out):
        #put the two layer together
        self.hidden_layer = HiddenLayer(Input,n_in,n_hidden,activation_function =T.tanh)
        #add logistic_layer on hidden layer
        self.logistic_layer = LogisticRegression(self.hidden_layer.output,n_hidden,n_out)
        #calculate the loss function
        self.loss = self.logistic_layer.get_loss
        #calculate the norm expression:L1_norm,L2_norm
        self.L1_norm = abs(self.hidden_layer.W).sum()+ abs(self.logistic_layer.W).sum() 
        self.L2_norm = (self.hidden_layer.W**2).sum()+(self.logistic_layer.W**2).sum()
        #collect all params used in gradient descent
        self.params = self.hidden_layer.params+self.logistic_layer.params
        #define final layer as handler
        self.Last_layer = self.logistic_layer
                

    def get_reg_cost(self,y,reg_l1,reg_l2):
        return self.loss(y) + reg_l1*self.L1_norm + reg_l2*self.L2_norm
        
    def save(self,model_name="MLP_model.pkl"):
        with open(model_name,'wb') as f:
            pickle.dump(self, f) 


'''the classical LeNet model for minist data-set'''
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
    


