
# YOUR CODE HERE
import numpy as np
class Dense:
    def __init__(self, N, H):
        self.W = np.random.randn(N, H)
        self.b = np.random.randn(H)
    def forward(self, X):
        Z = np.dot(X,self.W) + self.b
        self.X = X
        self.cache = locals()             
        return Z
    def backward(self, dZ):
        X = self.cache['X']        
        db = dZ.sum(axis=0)
        dX, dW = np.dot(dZ,np.transpose(self.W)), np.dot(np.transpose(X),dZ)
        return dX, dW, db

class Sigmoid:
    def forward(self, Z):
        H = 1 / (1 + np.exp(-Z))
        self.cache = locals()        
        return H
    def backward(self, dH):
        H = self.cache['H']        
        dZ = H * (1-H) * dH
        return dZ
    
class SoftmaxCE:
    def forward(self, S, Y):
        P = np.exp(S) / (np.exp(S).sum(axis=1, keepdims=True))
        y = Y.argmax(axis=1)
        M = len(P)
        L = P[np.arange(M), y]
        L = -np.log(L)
        L = np.expand_dims(L, axis=-1)
        self.cache = locals()
        return L
    def backward(self, dL):
        P, y, M = self.cache['P'], self.cache['y'], self.cache['M']
        dLdS = P
        dLdS[np.arange(M), y] -= 1
        dS = dLdS * dL
        return dS  