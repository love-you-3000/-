from 手写神经网络.Network import *
from 手写神经网络.mnist_loader import *
import datetime

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i][0] > max_value:
            max_value = vec[i][0]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = int(test_labels[i])
        predict = get_result(network.predict(test_data_set[i].tolist()))
        if label!= predict:
            error += 1
    return float(error) / float(total)

def train_and_evaluate(network):
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels,test_data_set, test_labels = load_data_wrapper()


    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print('%s epoch %d finished' % (datetime.datetime.now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    network = Network([784, 30, 10])
    train_and_evaluate(network)
    np.save("W1.npy",network.layers[0].W)
    np.save("b1.npy",network.layers[0].b)
    np.save("W2.npy",network.layers[1].W)
    np.save("b2.npy",network.layers[1].b)

