#coding=utf8
from Layers import Model_base
from theano import function,shared
from collections import namedtuple 
import theano.tensor as T
import numpy as np
import pickle
from PIL import Image
import pylab
import os
import collections
import lasagne
import time
import itertools,collections
import random
#get training data ready:
#TODO:pickle all image data,and classify training validate and testing data
data_type = collections.namedtuple("data_type", "feature label")

class Image_Info:
    channel = 1
    height = 64
    width = 64

def load_data(data_file=r'FER.data.pkl',indexFile=None,reload=False,
              img_dir=r"H:\ReasearchData\trainningData"):
    #if data still exists,then just load the pickle-file
    if reload is False and data_file is not None and os.path.exists(data_file):
        with open(data_file,'rb') as f:
            ret = pickle.load(f,encoding="bytes")
    else:
        print("reload all training data ...")
        if indexFile is None:
            indexFile = r"H:\ReasearchData\trainningData\indexFile\indexFile.txt"
        _f = lambda x :(os.path.join(img_dir,x[0]),x[-1])
        with open(indexFile,'r') as _file:
            f_contents = [_f(line.split()) for line in _file  ]
        random.shuffle(f_contents)#do random shuffle ——person dependent
        img_pool,labels = [],[]
        for img_path,label in f_contents:
            img = Image.open(open(img_path,'rb')).convert('L')
            #resize img
            img = img.resize((Image_Info.height, Image_Info.width), Image.ANTIALIAS)
#             pylab.figure(1)
#             pylab.imshow(img)
#             pylab.gray()
#             pylab.show()
#             a = input("asdasdasdasd:")
            #only contains gray channel
            img = np.asarray(img, dtype='float32') / np.float32(256)
#             print(img.shape)
            img_pool.append(img)
            labels.append(int(label))
        data_set = np.asarray(img_pool,dtype=np.float32).reshape(-1,Image_Info.channel,Image_Info.height,Image_Info.width)
        label_set = np.asarray(labels,dtype=np.int32)
        def get_split_data(data_set,label_set,traing_data_p=0.8,validate_data_p=0.1):
            assert len(data_set)==len(label_set)
            t1,t2 = int(len(data_set)*traing_data_p),int(len(data_set)*(traing_data_p+validate_data_p))
            return [(data_set[:t1],label_set[:t1]),(data_set[t1:t2],label_set[t1:t2]),(data_set[t2:],label_set[t2:])]
        ret =  get_split_data(data_set,label_set) 
        with open(r'FER.data.pkl','wb') as f:
            pickle.dump(ret,f,protocol=-1)
    
    train_data = data_type(*ret[0])
    validate_data = data_type(*ret[1])
    test_data = data_type(*ret[2])
    print("train_data:{} validate_data:{} test_data:{}".format(train_data[0].shape,
                         validate_data[0].shape,test_data[0].shape))
    return train_data,validate_data,test_data
        

#itertools function using mini-batch    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

     
     
        
#use lasagne to construct the neutral network
class NeutralNetwork(Model_base):
    
    def __init__(self):
        self.set_params()
        self.Input = T.tensor4("Input", dtype='float32')
        self.y = T.ivector("y")
        #learning_rate,to change them dynamically
        self.W_learning_rate = shared(value=np.float32(self.init_W_learning_rate),
                                    borrow = True,
                                    name='w_learning_rate')
        self.b_learning_rate = shared(value=np.float32(self.init_b_learning_rate),
                                    borrow = True,
                                    name='b_learning_rate')
        self.build_model()
        
        
    def set_params(self):
        #Learning Rate
        self.init_W_learning_rate = 0.08
        self.init_b_learning_rate = 0.4
        #initial Weight
#         self.init_conv_W = lasagne.init.Constant(0.05)
#         self.init_fully_W = lasagne.init.Constant(0.01)
        self.init_conv_W = lasagne.init.GlorotUniform()
        self.init_fully_W = lasagne.init.GlorotUniform()
        #Momentum
        self.momentom_w = 0.1
        self.momentom_b = 0.0
        #DropRate
        self.drop_out = 0.5
        #L2-Norm
        self.L2_norm = 0.001
    
    def build_model(self):
        #create Input layer
        
        input_layer = lasagne.layers.InputLayer(shape=(None, Image_Info.channel, Image_Info.height, Image_Info.width),
                                            input_var=self.Input,
                                            name = "input_layer")
        
        
        conv_layer1 = lasagne.layers.Conv2DLayer(input_layer,32,(5,5),
                                            W = self.init_conv_W,
                                            name = "conv_layer1")
        
        pool_layer1 = lasagne.layers.Pool2DLayer(conv_layer1,pool_size=(3,3),stride=(2,2),
                                                mode='max',ignore_border=True,
                                            name = "pool_layer1" )
        
        conv_layer2 = lasagne.layers.Conv2DLayer(pool_layer1,32,(5,5),
                                            W = self.init_conv_W,
                                            name = "conv_layer2")
        
        pool_layer2 = lasagne.layers.Pool2DLayer(conv_layer2,pool_size=(3,3),stride=(2,2),
                                                mode='max',ignore_border=True,
                                            name = "pool_layer2" )
        
        conv_layer3 = lasagne.layers.Conv2DLayer(pool_layer2,64,(5,5),
                                            W = self.init_conv_W,
                                            name = "conv_layer3")
        
        pool_layer3 = lasagne.layers.Pool2DLayer(conv_layer3,pool_size=(3,3),stride=(2,2),
                                                mode='max',ignore_border=True,
                                            name = "pool_layer3" )
        
        
        hidden_layer4 = lasagne.layers.DenseLayer(
                pool_layer3,
                num_units=1000,
                W = self.init_fully_W,
                nonlinearity=lasagne.nonlinearities.rectify,
                name = "hidden_layer4")
        
        hidden_layer5 = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(hidden_layer4, p=self.drop_out),
                num_units=1000,
                W = self.init_fully_W,
                nonlinearity=lasagne.nonlinearities.rectify,
                name = "hidden_layer5")
        
        #Attention:最后输出层千万不要加dropout
        last_layer = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(hidden_layer5, p=self.drop_out),
                num_units=6,
                W = self.init_fully_W,
                nonlinearity=lasagne.nonlinearities.softmax,
                name = "last_layer")
        
        self.layers = dict([
                            ("input_layer",input_layer),
                            ("conv_layer1",conv_layer1),
                            ("pool_layer1",pool_layer1),
                            ("conv_layer2",conv_layer2),
                            ("pool_layer2",pool_layer2),
                            ("conv_layer3",conv_layer3),
                            ("pool_layer3",pool_layer3),
                            ("hidden_layer4",hidden_layer4),
                            ("hidden_layer5",hidden_layer5),
                            ("last_layer",last_layer)])
        
        #individually get W,b
        self.params_W = lasagne.layers.get_all_params(self.layers['last_layer'],regularizable=True)
        self.params_b = lasagne.layers.get_all_params(self.layers['last_layer'],trainable=True,regularizable=False)
    #finish building the model
    
    @property
    def params(self):
        return self.params_W+self.params_b
    
    #used for update
    def cost(self,deterministic=False):
        return  lasagne.objectives.categorical_crossentropy(self.output(deterministic),self.y).mean()+self.L2_norm*lasagne.regularization.regularize_network_params(self.layers['last_layer'],
                              lasagne.regularization.l2, tags={'regularizable':True})
    
    def accuracy(self,deterministic=False):
        return T.mean(T.eq (T.argmax(self.output(deterministic), axis=1), self.y),dtype="float32")
    
    def output(self,deterministic=False):
        return lasagne.layers.get_output(self.layers['last_layer'],deterministic=deterministic)
    
    
    @property
    def update_function(self):
        grads = T.grad(self.cost(), self.params)
        updates = collections.OrderedDict()
        for i,(param, grad) in enumerate(zip(self.params, grads)):
            if i<len(self.params_W):
                updates[param] = param - self.W_learning_rate * grad
            else:
                updates[param] = param - self.b_learning_rate * grad
        #apply momentum to W ,b individually
        updates = lasagne.updates.apply_momentum(updates, params=self.params_W, momentum=self.momentom_w)
        updates = lasagne.updates.apply_momentum(updates, params=self.params_b, momentum=self.momentom_b)
        
        return updates
    

'''
save model params:
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this: 
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
'''
def save_model(network_last_layer,file_name='FER_PARAMS.npz'):
    np.savez(file_name,*lasagne.layers.get_all_param_values(network_last_layer))
     
def load_params(network_last_layer,file_name='FER_PARAMS.npz'):
    with np.load(file_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network_last_layer, param_values)



def train_process(batch_size = 100,#using stochastic gradient descent with mini-batch
                  epochs = 1000,#define how many times we pass the training data
                  validate_frequency = None,#validate data after how many patches we trained 
                  model_name="FER.pkl"
                  ):

    #loading training,validate,test data
    print("loading data...")
    training_data,validate_data,test_data = load_data()
    #train config
    n_train_batch = int(training_data.feature.shape[0] / batch_size) 
    n_validate_batch = int(validate_data.feature.shape[0] / batch_size) 
    n_test_batch = int(test_data.feature.shape[0] / batch_size)
    if validate_frequency is None:
        validate_frequency = n_train_batch
    '''compile:train,validate,test function  '''
    print("compiling  function...")
    model = NeutralNetwork()
    #compile train function
    train = function(
                     inputs = [model.Input,model.y],#训练时将特征和标签都作为输入
                     outputs = [model.accuracy(),model.cost()],
                     updates = model.update_function,
                     )
    #compile validate function    
    validate = function(
                     inputs = [model.Input,model.y],
                     outputs = model.accuracy(True),
                     )
    #conpile test function
    test = validate

    #begin training process
    best_accuracy = 0.0
    epoch  = 0
    patience = 10000
    patience_increase = 2
    accuracy_significant = 0.005
    stop_training = False
    best_test_accuracy = 0.0
    start_time = time.time()
    while epoch < epochs and not stop_training:
        epoch += 1
        #control learning rate decrease
        cur_W_learning_rate = model.W_learning_rate.get_value()
        cur_b_learning_rate = model.b_learning_rate.get_value()
        if cur_W_learning_rate>1e-3:
            model.W_learning_rate.set_value(np.float32(cur_W_learning_rate/(1+1e-3*epoch)))
        if cur_b_learning_rate>1e-3:
            model.b_learning_rate.set_value(np.float32(cur_b_learning_rate/(1+1e-3*epoch)))
        cur_W_learning_rate = model.W_learning_rate.get_value()
        cur_b_learning_rate = model.b_learning_rate.get_value()
        print("learning rate decrease:\nW_learning_rate:{}\tb_learning_rate:{}\n".format(cur_W_learning_rate,
                                         cur_b_learning_rate))
        #pass training data
        index = 0
        for Input,y in iterate_minibatches(training_data.feature, training_data.label, batch_size, shuffle=True):
            accuracy,loss = train(Input,y)
            print("trainning accuracy:{}\tloss:{}".format(accuracy,loss))
            passed_batches = (epoch-1)*n_train_batch + index+1
            index+=1
            if passed_batches%validate_frequency==0:
                #pass the validate data
                val_accuracy = np.mean([validate(Input,y) for Input,y in iterate_minibatches(validate_data.feature, 
                                    validate_data.label, batch_size)])
                print("epoch:{} learning_rate:W:{}\tb:{}\npass validate with validation_accuracy:{}  current iteration:{}/{}\n".format(
                            epoch,cur_W_learning_rate,cur_b_learning_rate,val_accuracy,passed_batches,min(patience,epochs*n_train_batch)))
                if val_accuracy > best_accuracy:#when get a better results
                    if val_accuracy >= best_accuracy*(1+accuracy_significant):
                        patience = max(patience,passed_batches*patience_increase)
                        print("significant progress has achieved,improve patients to:{}\n".format(patience))
                    best_accuracy = val_accuracy#update error 
                    #pass the test data
                    test_accuracy= np.mean([test(Input,y) for Input,y in iterate_minibatches(test_data.feature, 
                                    test_data.label, batch_size)])
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        print("model improves with test_accuracy:%{:2}".format(100*test_accuracy))
#                     save_model(model.layers['last_layer'],file_name="FER_PARAMS.npz")
                    model.save(model_name=model_name)
            if passed_batches>patience:
                stop_training = True
    end_time = time.time()
    print('finished training...\n costing {} seconds with test accuracy:%{:4}\nbest model has been saved...'.format(
                             end_time-start_time,100*best_test_accuracy))
       
    
    
emo_dict = [(0,u"NE"),(1,u"AN"),(2,u"DI"),(3,u"HA"),(4,u"SA"),(5,u"SU")]

def create_conf_Matrix(predict,truth,save_file = "cnn.txt"):
    save_file = open(save_file,'w+')
    global emo_dict
    assert len(predict)==len(truth)
    conf_matrix = {}
    for i,real_label in enumerate(truth):
        conf_matrix[(real_label,predict[i])]= conf_matrix.get((real_label,predict[i]),0)+1
    emo_dic = dict(emo_dict)
    print("\t",end='',file=save_file)
    for col in range(len(emo_dic)):
        print("\t{}".format(emo_dic[col]), end='',file=save_file)
    print("\tAccuracy",file=save_file)
    for row in range(len(emo_dic)):
        counter = 0
        print("\t{}".format(emo_dic[row]), end='',file=save_file)
        for col in range(len(emo_dic)):
            cur = conf_matrix.get((row,col),0)
            counter += cur
            print("\t{}".format(cur), end='',file=save_file)
        print("\t%{:.4}".format(100*conf_matrix[(row,row)]/counter),file=save_file)
    save_file.close()

def load_model(path='FER.pkl'):
    if not os.path.exists(path):
        print("invalid model path")
        return
#     ("input_layer",input_layer),
#     ("conv_layer1",conv_layer1),
#     ("pool_layer1",pool_layer1),
#     ("conv_layer2",conv_layer2),
#     ("pool_layer2",pool_layer2),
#     ("conv_layer3",conv_layer3),
#     ("pool_layer3",pool_layer3),
#     ("hidden_layer4",hidden_layer4),
#     ("hidden_layer5",hidden_layer5),
#     ("last_layer",last_layer)])
    with open(path,'rb') as f:
        model = pickle.load(f,encoding="bytes")
    #compile predict function
    predict = function(inputs=[model.Input],
                       outputs = lasagne.layers.get_output(model.layers['pool_layer2'],deterministic=True),
                       )
    #loading training,validate,test data
    print("loading data...")
    _,_,test_data = load_data()
    results = predict(test_data.feature)
    print(results)
#     visualize.plot_conv_activity(layer, x, figsize)
#     create_conf_Matrix(results,test_data.label)



if __name__=="__main__":
#     train_process()
    load_model()




