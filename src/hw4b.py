"""
Source Code for Homework 4.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

"""

import numpy

import theano
from theano import tensor as T

from hw4_utils import shared_dataset
from hw4_nn import myMLP, train_nn

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(nbit,num))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    return X,Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    pass

#TODO: build and train a MLP to learn parity function
def test_mlp_parity():
    # generate datasets
    train_set = gen_parity_pair(8, 1000)
    valid_set = gen_parity_pair(8, 500)
    test_set  = gen_parity_pair(8, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

#TODO: build and train a RNN to learn parity function
def test_rnn_parity():
    pass

if __name__ == '__main__':
    test_mlp_parity()
