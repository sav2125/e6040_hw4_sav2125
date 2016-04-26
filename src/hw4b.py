"""
Source Code for Homework 4.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

"""

import numpy
from collections import OrderedDict
from itertools import product

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
    X = numpy.random.randint(2, size=(num,nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    return X,Y

#TODO: implement RNN class to learn parity function

class RNNSLU(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nout = 2, nhidden = 10, ninput = 10):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model

        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (ninput, nhidden))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nhidden, nhidden))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nhidden, nout))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nhidden,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nout,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nhidden,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = T.imatrix()
        # x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        # y_sentence = T.ivector('y_sentence')  # labels


        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

        


def test_rnnslu_parity(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    # process input arguments
    param = {
        'fold': 3,
        'lr': 0.5,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 200,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 40,
        'savemodel': False,
        'normal': True,
        'folder':'../result'}
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    #check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')
    #train_set, valid_set, test_set, dic = load_data(param['fold'])
    train_set = gen_parity_pair(8, 1000)
    valid_set = gen_parity_pair(8, 500)
    test_set  = gen_parity_pair(8, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    

    # create mapping from index to label, and index to word
    # unpack dataset
    train_lex, train_y = train_set
    valid_lex, valid_y = valid_set
    test_lex, test_y = test_set

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    rnn = RNNSLU(
        nout=2,
        nhidden=param['nhidden'],
        ninput=8
    )

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        # shuffle
        shuffle([train_lex, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            rnn.train(x, y, param['win'], param['clr'])
            #print('[learning] epoch %i >> %2.2f%%' % (
            #    e, (i + 1) * 100. / nsentences), end=' ')
            #print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')
            sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: idx2label[x],
                            rnn.classify(numpy.asarray(
                            contextwin(x, param['win'])).astype('int32')))
                            for x in test_lex]
        predictions_valid = [map(lambda x: idx2label[x],
                             rnn.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32')))
                             for x in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             param['folder'] + '/current.test.txt',
                             param['folder'])
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])

        if res_valid['f1'] > best_f1:

            if param['savemodel']:
                rnn.save(param['folder'])

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid['f1']

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = e

        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])

#TODO: build and train a MLP to learn parity function
def test_mlp_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False,inp = 8):
    # generate datasets
    train_set = gen_parity_pair(inp, 1000)
    valid_set = gen_parity_pair(inp, 500)
    test_set  = gen_parity_pair(inp, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    # classifier = myMLP(...)
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=inp,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=2
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)


#TODO: build and train a RNN to learn parity function

if __name__ == '__main__':
    test_mlp_parity()
