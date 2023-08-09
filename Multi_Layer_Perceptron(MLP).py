"""
Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

import numpy as np
import os
import sys

sys.path.append('Utils')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.9, num_bn_layers=1):

        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        self.linear_layers = []#[Linear(self.input_size, self.output_size, weight_init_fn, bias_init_fn)]
        self.bn_layers = []
        if len(hiddens) == 0:
            self.linear_layers = [Linear(self.input_size, self.output_size, weight_init_fn, bias_init_fn)]
            if self.bn and i < self.num_bn_layers:
                self.bn_layers.append(BatchNorm(self.output_size, alpha=0.9))
        else:
            for i in range(self.nlayers):
                if i == 0:
                    self.linear_layers.append(Linear(self.input_size, hiddens[i], weight_init_fn, bias_init_fn))
                    if self.bn and i < self.num_bn_layers:
                        self.bn_layers.append(BatchNorm(hiddens[i], alpha=0.9))
                elif i == len(hiddens):
                    self.linear_layers.append(Linear(hiddens[i-1], output_size, weight_init_fn, bias_init_fn))
                    if self.bn and i < self.num_bn_layers:
                        self.bn_layers.append(BatchNorm(output_size, alpha=0.9))
                else:
                    self.linear_layers.append(Linear(hiddens[i-1], hiddens[i], weight_init_fn, bias_init_fn))
                    if self.bn and i < self.num_bn_layers:
                        self.bn_layers.append(BatchNorm(hiddens[i], alpha=0.9))
            

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        # if self.bn:
            # for i in range(self.num_bn_layers):
                # if i ==0:
                    # self.bn_layers.append(BatchNorm(self.input_size, alpha=0.9))
                # else:
                    # self.linear_layers.append(Linear(hiddens[i-1], hiddens[i], weight_init_fn, bias_init_fn))
        # else:    
            # self.bn_layers = None #[BatchNorm(self.input_size, alpha=0.9)]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through the entire MLP.
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i].forward(x)
            if self.bn and i < self.num_bn_layers:
                x = self.bn_layers[i].forward(x,not self.train_mode)
            x = self.activations[i].forward(x)
        self.out = x
        return x

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out the backpropped derivatives in each
        # of the linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
        
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        #self.linear_layer[0].W 
        
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].momentum_W = self.momentum*self.linear_layers[i].momentum_W - self.lr*self.linear_layers[i].dW
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            self.linear_layers[i].momentum_b = self.momentum*self.linear_layers[i].momentum_b - self.lr*self.linear_layers[i].db
            self.linear_layers[i].b += self.linear_layers[i].momentum_b
            # self.linear_layers[i].W -= self.lr*self.linear_layers[i].dW
            # self.linear_layers[i].b -= self.lr*self.linear_layers[i].db
          #  pass
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                # Update weights and biases here
               self.bn_layers[i].gamma += -self.lr*self.bn_layers[i].dgamma
               self.bn_layers[i].beta += -self.lr*self.bn_layers[i].dbeta

        #raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        L = self.criterion(self.out,labels)
        dL = self.criterion.derivative()
        for i in range(len(self.linear_layers)-1, -1, -1):
            dL = dL*self.activations[i].derivative()
            
            if self.bn and i < self.num_bn_layers:
                dL = self.bn_layers[i].backward(dL)
            dL = self.linear_layers[i].backward(dL)
            

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    
