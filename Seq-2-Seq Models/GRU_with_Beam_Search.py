import numpy as np
import sys

sys.path.append('Utils')
from gru_cell import *
from linear import *

# This is the neural net that will run one timestep of the input
# This is to test that the GRU Cell implementation is correct when used as a GRU.
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # The network consists of a GRU Cell and a linear layer
        self.rnn = GRU_Cell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)

    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        # A pass through one time step of the input

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.h = h
        self.hidden = self.rnn(self.x, self.h)
        self.logits = self.projection(self.hidden)
        return self.logits, self.hidden

# An instance of the class defined above runs through a sequence of inputs to
# generate the logits for all the timesteps.
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]
    t = 0
    seq_len = inputs.shape[0]
    hidden = np.zeros((net.hidden_dim))
    logits = np.zeros((seq_len,net.num_classes))
    
    for ip in inputs:
        output, hidden = net.forward(ip, hidden)
        logits[t] = output
        t += 1
    
    return logits

