import pickle as cPickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    # x是一个长度为784的数组，将它改变成784*1的数组
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 将一维的y改变成10维的数组，其中第y维为1，其余维为0
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # training_data =[(training_inputs[0],training_results[0]),...()]
    training_data = list(zip(training_inputs, training_results))
    # 验证数据集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    # 测试数据集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_inputs,training_results, test_inputs,te_d[1])

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def return_valid():
    tr_d, va_d, te_d = load_data()
    # x是一个长度为784的数组，将它改变成784*1的数组
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 将一维的y改变成10维的数组，其中第y维为1，其余维为0
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # training_data =[(training_inputs[0],training_results[0]),...()]
    training_data = list(zip(training_inputs, training_results))
    # 验证数据集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    # 测试数据集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return validation_inputs,va_d[1]

