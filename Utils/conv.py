import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """ 
        batch, inc, inp = x.shape
        self.Dim = int(np.floor((inp - self.kernel_size)/self.stride)+1)
        out = np.zeros((batch,self.out_channel,self.Dim))
        self.x = x
        for i in range(batch):
            for j in range(self.out_channel):
                for k in range(self.Dim):
                    out[i,j,k] = np.sum(self.W[j,:,:]*x[i,:, self.stride*k:self.stride*k + self.kernel_size]) + self.b[j]
                    #x[i,:, self.stride*k:self.stride*k + self.kernel_size]*self.W[j,:,:]
                    #
                #out[i,j,k] += self.b[j]
        
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch, outc, outp = delta.shape
        dx = np.zeros((batch,self.in_channel,self.x.shape[2]))
        for i in range(batch):
            for j in range(outp):
                for k in range(self.out_channel):
                    self.dW[k,:,:] += delta[i,k,j]*self.x[i,:, self.stride*j:self.stride*j + self.kernel_size]
                    self.db[k] += delta[i,k,j]*1
                    dx[i,:, self.stride*j:self.stride*j + self.kernel_size] += self.W[k,:,:]*delta[i,k,j]    
        return dx 



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        #for i in range():
        out = x.reshape(self.b,-1)
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = delta.reshape(self.b,self.c,self.w)
        return dx
