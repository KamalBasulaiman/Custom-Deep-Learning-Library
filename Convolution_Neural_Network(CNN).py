import numpy as np
import os
import sys

sys.path.append('Utils')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):

    """
    A simple convolutional neural network

    Here we build implement the same architecture described in Section 3.3
    we need to specify the detailed architecture in function "get_cnn_model" below
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        We can do sanity check for len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """
        
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------
        ## The code goes as -->
        # self.convolutional_layers (list Conv1D) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        self.convolutional_layers = []
        self.output_size = []
        self.flatten = [Flatten()]
        
        
        for i in range(self.nlayers):
            if i ==0:
                self.convolutional_layers.append(Conv1D(num_input_channels, num_channels[i], kernel_sizes[i], strides[i], conv_weight_init_fn, bias_init_fn))
                self.output_size.append(((input_width - kernel_sizes[i])//strides[i])+1)
            else:
                self.convolutional_layers.append(Conv1D(num_channels[i-1], num_channels[i], kernel_sizes[i], strides[i], conv_weight_init_fn, bias_init_fn))
                self.output_size.append(((self.output_size[i-1] - kernel_sizes[i])//strides[i])+1)

        
        self.linear_layer = [Linear(self.output_size[-1]*num_channels[-1], num_linear_neurons, linear_weight_init_fn, bias_init_fn)]
        
    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """
        # Iterate through each layer
        # <---------------------
        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i].forward(x)
            x = self.activations[i].forward(x)
            
        x = self.flatten[0].forward(x)
        x = self.linear_layer[0].forward(x)
        # Save output (necessary for error and loss)
        self.output = x

        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()
        grad = self.linear_layer[0].backward(grad)
        grad = self.flatten[0].backward(grad)
        for i in range(len(self.convolutional_layers)-1, -1, -1):
            grad *= self.activations[i].derivative()
            grad = self.convolutional_layers[i].backward(grad)
        # Iterate through each layer in reverse order
        # <---------------------

        return grad


    def zero_grads(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
