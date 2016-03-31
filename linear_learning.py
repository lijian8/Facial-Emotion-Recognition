#coding=utf8
import random
'''
采用梯度下降法，对单变量进行回归，拟合单变量曲线
training data:
x    y = 2*x+14
1    16
2    18
3    20
4    22
5    24
6    26
7    28
..
损失函数；
cost_function = 1/(2M)* Acc{(theta0 + theta1*x - y)^2}
损失函数的梯度(该变量使得损失函数增加最快的方向)_注意这里:损失函数的方向与训练样本无关，因此可以沿着这个方向一直下降到最低点(只有2个样本也可以训练)
d_theta0：1/m*ACC{theta0+theta1*x -y}
d_theta1：1/m*ACC{[theta0+theta1*x -y]*x}
参数更新:
theta0 := theta0 - alpha* d_theta0
theta1 := theta1 - alpha* d_theta1
损失函数的阈值:
10-5
''' 
def calculate_cost_function(training_data,theta0,theta1):
    cost = 0.0
    M = len(training_data)
    for x,y in training_data:
#         print("theta0:{} theta1:{} x:{} y:{}".format(theta0,theta1,x,y))
        cost += pow(float(theta0 + theta1*x - y),2)/(2.*M)
    return cost
#do batch gradient update
def update_theta(theta0,theta1,theta2,theta3,training_data,learning_rate = 1.0):
    d_theta0,d_theta1 = 0.0,0.0
    for x,y in training_data:
        d_theta0 += (theta0 + theta1*x-y)
        d_theta1 += ((theta0 + theta1*x-y)*x)
    d_theta0,d_theta1 = d_theta0/len(training_data),d_theta1/len(training_data)
    theta0 = theta0 - learning_rate* d_theta0
    theta1 = theta1 - learning_rate* d_theta1
    return theta0,theta1

    
training_data = [(i/1000.0,2*i/1000.0+12+random.random()) for i in range(1,1000)]
M = len(training_data)
theta0 = 19000
theta1 = 23333
cost = 10000
print("initialize: theta0 : {}\t theta1 : {}".format(theta0,theta1))
while cost>=1e-5:
    for index,traing_sample in enumerate(training_data):
        theta0,theta1 = update_theta(traing_sample[0],traing_sample[1],theta0,theta1,training_data)
        cost = calculate_cost_function(training_data,theta0,theta1)
        print("update using sample:{}\ntheta0:{} \ttheta1:{} \tloss:{}".format(traing_sample,theta0,theta1,cost))
