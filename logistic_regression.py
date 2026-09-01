import numpy as np
class logisitic_regressionScratch:
    def __init__(self,learning_rate = 0.01,n_iterations = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def compute_cost(self,X,y):
        m = X.shape[0]
        z = np.dot(X,self.w)+ self.b
        y_pred = self.sigmoid(z)
        cost = -(1/m) * np.sum(y*np.log(y_pred) + (1-y) *np.log(1-y_pred))
        return cost
    def compute_gradient(self,X,y):
        m = X.shape[0]
        z = np.dot(X,self.w)+ self.b
        y_pred = self.sigmoid(z)
        dw = (1/m) * np.dot(X.T,y_pred-y)
        db = (1/m) * np.sum(y_pred-y)
        return dw,db
    def fit(self,X,y):
        n_samples ,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for i in range(self.n_iterations):
            dw,db = self.compute_gradient(X,y)
            self.w = self.w - dw*self.lr
            self.b = self.b - db * self.lr
