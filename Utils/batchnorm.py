import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
        if eval:
            self.mean = self.running_mean
            self.var = self.running_var
            self.norm = (x-self.running_mean)/(np.sqrt(self.running_var+self.eps))
            self.out = self.gamma*self.norm+self.beta
        else:
            self.x = x
            self.mean = np.mean(x, axis = 0)
            self.var = np.var(x, axis = 0)
            self.norm = (x-self.mean)/(np.sqrt(self.var+self.eps))
            self.out = self.gamma*self.norm+self.beta       
            # Update running batch statistics
            self.running_mean = self.alpha*self.running_mean + (1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var + (1-self.alpha)*self.var
            
        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        self.dgamma = np.sum(delta*self.norm, axis = 0, keepdims = True)
        self.dbeta = delta.sum(axis = 0, keepdims= True)
        self.dnorm = delta*self.gamma
        self.dvar =  -0.5*np.sum(self.dnorm*(self.x-self.mean)*np.power(self.var+self.eps,-1.5), axis = 0)
        self.dmean = -np.sum(self.dnorm*(np.power(self.var+self.eps,-0.5)), axis = 0) - 2/(delta.shape[0]) * self.dvar * np.sum(self.x - self.mean, axis = 0)
        out = self.dnorm*(np.power(self.var+self.eps,-0.5)) + 2/(delta.shape[0])*self.dvar*(self.x-self.mean) + self.dmean*(1/delta.shape[0])
        return out
