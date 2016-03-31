#coding=utf8
import theano.tensor as T
from theano import shared,function
import numpy as np
from logistic_regression import LogisticRegression,load_data
import pickle



def  rectify_linear_activate(x):
    return T.maximum(x, 0.0)

 
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
        
        
class MLP(object):
    "the MLP object contains a hidden layer and a softmax-classifier"
    def __init__(self,Input,n_in,n_hidden,n_out):
        #put the two layer together
        self.hidden_layer = HiddenLayer(Input,n_in,n_hidden,activation_function =rectify_linear_activate)
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
        
        

def train_process(cost_norm_reg_l1=0.00,cost_norm_reg_l2=0.001,#used for L1(or L2)-norm regulation 
                  learning_rate=0.13,
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
    #compile train function
    x = T.fmatrix('x')
    y = T.ivector('y')
    index  = T.lscalar('index')
    #set n_in = 28*28 n_hidden = 500,n_out = 10,reg_l1 = 0.00 reg_l2 = 0.001
    mlp = MLP(x,28*28,500,10)
    cost = mlp.get_reg_cost(y, cost_norm_reg_l1,cost_norm_reg_l2)
    t_params = T.grad(cost,mlp.params)
    updates = [(param,param-learning_rate*t_param) for (param,t_param) in
              zip(mlp.params,t_params)]
    train = function(inputs = [index],
             outputs = [mlp.Last_layer.get_errors(y)],
             updates = updates,
             givens = [
                       (x,training_data.feature[index*batch_size:(index+1)*batch_size]),
                       (y,training_data.label[index*batch_size:(index+1)*batch_size]),]
             )
    #compile validate function
    validate = function(inputs = [index],
             outputs = [mlp.Last_layer.get_errors(y)],
             givens = [
                       (x,validate_data.feature[index*batch_size:(index+1)*batch_size]),
                       (y,validate_data.label[index*batch_size:(index+1)*batch_size]),]
             )
    #conpile test function
    test = function(inputs = [index],
             outputs = [mlp.Last_layer.get_errors(y)],
             givens = [
                       (x,test_data.feature[index*batch_size:(index+1)*batch_size]),
                       (y,test_data.label[index*batch_size:(index+1)*batch_size]),]
             )   
    #begin training process
    best_error = np.inf
    epoch  = 0
    patience = 10000
    patience_increase = 2
    error_significant = 0.01
    stop_training = False
    while epoch < epochs and not stop_training:
        epoch += 1
        for index in  range(n_train_batch):
            error = train(index)
#             print('error:{}'.format(error))
            passed_batches = (epoch-1)*n_train_batch + index+1
            if passed_batches%validate_frequency==0:
                #pass the validate data
                val_error = np.mean([validate(i) for i in range(n_validate_batch)])
                print("pass validate with validation_error:{}  current iteration:{}/{}".format(
                            val_error,passed_batches,min(patience,epochs*n_train_batch)))
                if val_error < best_error:#when get a better results
                    if val_error <= best_error*(1-error_significant):
                        patience = max(patience,passed_batches*patience_increase)
                    best_error = val_error#update error 
                    #pass the test data
                    test_error = np.mean([test(i) for i in range(n_test_batch)])
                    print("model improves with test accuray:%{:2}".format(100*(1-test_error)))
                    mlp.save()
            if passed_batches>patience:
                stop_training = True
                    


if __name__=="__main__":
    train_process(cost_norm_reg_l1=0.00,cost_norm_reg_l2=0.0001)
#     rectify_linear_activate(2)

        
        
        
        
        
        
        
