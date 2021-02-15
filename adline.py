import numpy as np

class AdlineGD(object):
    def __init__(self, eta=0.0001, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = 1
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1] + 1)
        
        for i in range(self.n_iter):
            predict = self.activation(np.dot(x, self.w[1:]) + self.w[0])
            errors = y - predict
            self.w[0] += self.eta * errors.sum()
            self.w[1:] += self.eta * np.dot(x.T, errors)
        
        return self

    def activation(self, x):
        return x
    
    def forward(self, x):
        predict = np.dot(x, self.w[1:]) + self.w[0]
        return np.where(self.activation(predict) >=0, 1, -1)
