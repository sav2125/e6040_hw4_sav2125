# e6040_hw4_sav2125

You are asked to experiment with recurrent neural networks and input embeddings. Start by going through the Deep Learning Tutorials Project, especially, the Theano scan function and RNN [1, 2, 3]. The source code provided in the Homework 4 repository is excerpted from rnnslu.py.
You will be using the Airline Travel Information System (ATIS) dataset. A python routine called load data is provided to you for downloading and preprocessing the dataset. You should use it, unless you have absolute reason not to. The ﬁrst time you call load data, it will take you some time to download the dataset.

Here, you are asked to revisit the universal approximation theory. In particular, implement a shallow MLP (2-layer), a deep MLP, and a RNN to learn the parity function. Although the universal approximation theory guarantees that the parity function can be learned by a neural network, a 2-layer MLP (one hidden layer) might need an enormous number of hidden neurons (on the order of magnitude of an expo-nential in the number of input bits). In contrast, a deep MLP requires signiﬁcantly lower number of neurons and a RNN requires even less.
