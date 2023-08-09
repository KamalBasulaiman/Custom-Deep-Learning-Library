import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step (hidden dim)

        self.x = x
        self.hidden = h

        self.r = self.r_act.forward(np.tensordot(self.Wrx,self.x,1) + np.tensordot(self.Wrh,self.hidden,1))
        self.z = self.z_act.forward(np.tensordot(self.Wzx,self.x,1) + np.tensordot(self.Wzh,self.hidden,1))
        self.h_tilda = self.h_act.forward(np.tensordot(self.Wh,(self.r*self.hidden),1) + np.tensordot(self.Wx,self.x,1))
        self.h_t = ((1-self.z)*self.hidden) + (self.z*self.h_tilda)
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        #assert self.x.shape == (self.d, )
        #assert self.hidden.shape == (self.h, )

        #assert self.r.shape == (self.h, )
        #assert self.z.shape == (self.h, )
        #assert self.h_tilda.shape == (self.h, )
        #assert h_t.shape == (self.h, )

        # return h_t
        return self.h_t


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.


        self.x = self.x.reshape(1,-1)
        self.z = self.z.reshape(1,-1)
        self.r = self.r.reshape(1,-1)
        self.hidden = self.hidden.reshape(1,-1)
        self.h_tilda = self.h_tilda.reshape(1,-1)
   
        dh_tilda = delta * self.z
       
        self.dWx += np.tensordot((dh_tilda*self.h_act.derivative().reshape(1,-1)).T, self.x,1)
        self.dWh += np.tensordot((dh_tilda*self.h_act.derivative().reshape(1,-1)).T,self.r*self.hidden,1)
        self.dWrx += np.tensordot((np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wh,1)*self.hidden*self.r_act.derivative().reshape(1,-1)).T,self.x,1)
        self.dWrh += np.tensordot((np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wh,1)*self.hidden*self.r_act.derivative().reshape(1,-1)).T,self.hidden,1)
        self.dWzx += np.tensordot((delta*(self.h_tilda-self.hidden)*self.z_act.derivative().reshape(1,-1)).T,self.x,1)
        self.dWzh += np.tensordot((delta*(self.h_tilda-self.hidden)*self.z_act.derivative().reshape(1,-1)).T,self.hidden,1)
        
        # 2) Compute dx, dh
        dh = delta*(1-self.z) + \
            np.tensordot(delta*(self.h_tilda-self.hidden)*self.z_act.derivative().reshape(1,-1),self.Wzh,1) + \
            np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wh,1)*self.r + \
            np.tensordot(np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wh,1)*self.hidden*self.r_act.derivative().reshape(1,-1),self.Wrh,1)
           
        dx = np.tensordot(np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wh,1)*self.hidden*self.r_act.derivative().reshape(1,-1),self.Wrx,1) + np.tensordot(dh_tilda*self.h_act.derivative().reshape(1,-1),self.Wx,1) + np.tensordot(delta*((self.h_tilda-self.hidden)*self.z_act.derivative()).reshape(1,-1),self.Wzx,1)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
