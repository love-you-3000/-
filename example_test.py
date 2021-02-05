import numpy as np
from 手写神经网络.Network import *
from 手写神经网络.test import *
from 手写神经网络.mnist_loader import *

test_data_set,test_labels = return_valid()
w1 = np.load("W1.npy")
w2 = np.load("W2.npy")
b1 = np.load("b1.npy")
b2 = np.load("b2.npy")

network = Network([784, 30, 10])

network.layers[0].W = w1
network.layers[0].b = b1
network.layers[1].W = w2
network.layers[1].b = b2

right = 0
for i in range(len(test_data_set)):
    network.layers[0].forward(test_data_set[i])
    network.layers[1].forward(network.layers[0].output)
    if get_result(network.layers[1].output)==test_labels[i]:
        right+=1
rate = (right/len(test_data_set))*100
print("正确率:"+str(rate)+"%")
