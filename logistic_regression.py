import numpy as np
class logisitic_regressionScratch:
    def __init__(self,learning_rate = 0.01,n_iterations = 1000,regularization = None, lambda_ = 0.1):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.regularization = regularization
        self.lambda_ = lambda_
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def compute_cost(self,X,y):
        m = X.shape[0]
        z = np.dot(X,self.w)+ self.b
        y_pred = self.sigmoid(z)
        cost = -(1/m) * np.sum(y*np.log(y_pred) + (1-y) *np.log(1-y_pred))
        if(self.regularization == 'L1'):
            cost = cost + (self.lambda_/(2*m)) * np.sum(self.w**2)
        elif(self.regularization == 'L2'):
            cost = cost + (self.lambda_/(m)) * np.sum(self.w)
        return cost
    def compute_gradient(self,X,y):
        m = X.shape[0]
        z = np.dot(X,self.w)+ self.b
        y_pred = self.sigmoid(z)
        dw = (1/m) * np.dot(X.T,y_pred-y)
        db = (1/m) * np.sum(y_pred-y)
        if(self.regularization == 'L1'):
            dw = dw + (self.lambda_/m) * np.sign(self.w)
        elif(self.regularization == 'L2'):
            dw = (self.lambda_/m)*self.w
        return dw,db
    def fit(self,X,y):
        n_samples ,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for i in range(self.n_iterations):
            dw,db = self.compute_gradient(X,y)
            self.w = self.w - dw*self.lr
            self.b = self.b - db * self.lr
    def predict_proba(self,X):
        z =np.dot(X,self.w)+ self.b 
        return self.sigmoid(z)       
    def predict(self,X,threshold = 0.5):
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
