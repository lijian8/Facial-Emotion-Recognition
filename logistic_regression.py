#coding=utf8
from theano import function,shared
from collections import namedtuple 
import theano.tensor as T
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
    def __init__(self,input,n_in,n_out):
        #weights always use shared variables:W,b
        #特别注意：权重变量和偏移变量 shared的类型是否为float32,决定了整个程序能否运行在gpu上！！！
        self.W = shared(np.asarray(0.001*np.random.random((n_in,n_out)),dtype='float32'),
                        name = "LogisticRegression_W",
                        borrow = True)
     
        self.b = shared(np.asarray(0.001*np.random.random((n_out,)),dtype='float32'),
                        name = "LogisticRegression_b",
                        borrow = True)
        self.output_matrix = T.exp(T.dot(input,self.W)+self.b)
         
        self.p_y_given_x,self.output = LogisticRegression.do_softMax(self.output_matrix)
        #保存变量
        self.params = [self.W,self.b]
        self.input = input#之所以要加这个，是为了方便以后直接修改输入矩阵就可以预测

        
        
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


def training_process():
    #加载训练，验证和测试数据集
    training_data,validate_data,test_data = load_data(r"mnist.pkl.gz")
    #stochastic gradient descent configuration
    learning_rate = 0.03 #学习率
    batch_size = 600   #每次取多少个样本来决定下降的梯度
    '''注意：这里需要将变量转换为numpy.ndarray '''
    n_train_batches = int(training_data.feature.get_value().shape[0]/batch_size)
    n_valid_batches = int(validate_data.feature.get_value().shape[0]/batch_size)
    n_test_batches = int(test_data.feature.get_value().shape[0]/batch_size)
    print("n_train_batches:{}".format(n_train_batches))
    epochs = 100     #优化过程最多过几遍数据集
    validation_frequency = n_train_batches #训练多少个batch之后跑一遍验证集，默认为n_train_batches,
    #如果验证集的正确率有提高，那么更新模型，并在测试集上跑一遍；在验证集上正确率没有提升，继续训练下一个batch
    best_validation_rate = np.inf#记录在验证集上最好的正确率
    
    
    '''编译训练模型所需要的函数'''
    x_matrix = T.fmatrix('X')#输入一个batch的训练样本矩阵
    y_label = T.ivector('y')#输入训练样本的分类
    classifier = LogisticRegression(x_matrix,training_data.feature.get_value().shape[1],10)#28*28 是训练集中每一行的特征维数，10是输出的分类个数
    cost = classifier.get_loss(y_label)#损失函数的变量表达式
    error = classifier.get_errors(y_label)#错误率的变量表达式
    updates = classifier.get_updates(y_label,learning_rate=learning_rate)
    
    
    index = T.lscalar("index") #batch的索引号
    '''编译在训练集上一个batch训练样本的训练函数'''
    train = function(inputs=[index],
              outputs= error,
              updates = updates,
              givens= [#这里之所以出错是因为，data是np.ndarray类型,而不是tensor variables
                     (x_matrix,training_data.feature[index*batch_size:(index+1)*batch_size]),
                     (y_label,training_data.label[index*batch_size:(index+1)*batch_size])],)
    '''编译在验证集上一个batch训练样本的训练函数：与训练集上的区别仅在于只是单纯的分类，不会更新权值'''
    validate = function(inputs=[index],
              outputs= error,
              givens= [
                     (x_matrix,validate_data.feature[index*batch_size:(index+1)*batch_size]),
                     (y_label,validate_data.label[index*batch_size:(index+1)*batch_size])],)
    '''编译在测试集上一个batch训练样本的训练函数'''
    test = function(inputs=[index],
              outputs= error,
              givens= [
                     (x_matrix,test_data.feature[index*batch_size:(index+1)*batch_size]),
                     (y_label,test_data.label[index*batch_size:(index+1)*batch_size])],)
    
    epoch = 0 
    while epoch <= epochs:
        epoch +=1
        #每个epoch将所有训练集过一遍
        for batch_index in range(n_train_batches):
            #对于每个batch_index,do training
            error = train(batch_index)
            iter = (epoch-1) * n_train_batches + batch_index
            if iter%validation_frequency ==0:#循环次数到达验证次数，对验证集进行验证
#                 print("iter:{}".format(iter))
                val_error = [validate(i) for i in range(n_valid_batches)]
                validation_error = np.mean(val_error)
                print("validation_error:{}".format(validation_error))
                if validation_error < best_validation_rate:#当前模型比最好的还要好
                    best_validation_rate = validation_error
                    #在测试集上测试当前模型
                    t_error = [test(i) for i in range(n_test_batches)]
                    test_error = np.mean(t_error)
                    print("current model make a break with\nvalidation error:{}\t test error:{}".format(validation_error,test_error))
                    #TODO:save the model
                    classifier.save()
    


'''
Attention:在处理图片之前需要将图片数据先/255个灰度值，归一化到0-1之间
'''
def test_model(model_path ="Logistic_model.pkl"):
    #加载训练，验证和测试数据集
    training_data,validate_data,test_data = load_data(r"mnist.pkl.gz")
    #加载训练好的模型
    classifier = pickle.load(open(model_path,'rb'),encoding="bytes") 
    '''特别注意：为了使用pickle load的模型，必须使用将所有对象方法返回表达式中的变量都指定作为输入，但是
            因为我们初始化对象时，传入的参数就是推演对象表达式所需的所有数据，但load之后我们无法再次对对象进行初始化，因此
            需要将初始化参与表达式计算的所有变量都保存起来，编译函数时直接给其赋值，这样我们就可以不用调初始化函数就可以复用
            一个模型
    '''
    # compile a predictor function
    predict_model = function(
        inputs=[classifier.input],#注意这里
        outputs=classifier.output)
    # compile a loss function
    y = T.ivector("label")
    loss = function(
                inputs = [classifier.input,y],#注意这里
                outputs = classifier.get_errors(y),#需要知道所有表达式中的变量，一直推到input
                allow_input_downcast=True
                )
    
    features = test_data.feature.get_value()[:1]#feature 为float32 shared 在gpu显存中
    labels = test_data.label.eval()[:1]#labels 由shared转为int 之后，无法保存于gpu显存中，保存在内存里，
    #是普通的tensor variable,注意：tensor variable取值的方法
    print("feature:\n{}".format(features))
    
    pred = predict_model(features)
    los = loss(features,labels)  
    print("type:{}".format(type(labels)))
    print("predict labels:\n{} loss:%{}\n ".format(pred,los*100))
    
    
    
    
    

if __name__=="__main__":
#     training_process()
    test_model("Logistic_model.pkl")
