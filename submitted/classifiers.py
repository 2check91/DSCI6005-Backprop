
# YOUR CODE HERE
import numpy as np
import layers
       
class LayeredMLP():
    def __init__(self, nb_feature, nb_hidden, nb_class):        
        self.dense1 = layers.Dense(N=nb_feature, H=nb_hidden)
        self.sigmoid = layers.Sigmoid()
        self.dense2 = layers.Dense(N=nb_hidden, H=nb_class)
        self.softmaxce = layers.SoftmaxCE()
        
    def _f(self, X):
        Z = self.dense1.forward(X) 
        H = self.sigmoid.forward(Z)
        S = self.dense2.forward(H)        
        self.cache = locals()
        
    def predict(self, X):
        self._f(X)
        S = self.cache['S']
        P = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
        return P

    def evaluate(self, X, Y):
        return np.mean((self.predict(X)).argmax(axis=1) == Y.argmax(axis=1))  
    
    def forward(self, X, Y):
        self._f(X)
        S = self.cache['S']
        L = self.softmaxce.forward(S, Y)        
        return np.mean(L)    
# YOUR CODE HERE
class LayeredMLPWithGDOptimizer(LayeredMLP):
    def __init__(self, nb_feature, nb_hidden, nb_class, alphalr=0.01):        
        LayeredMLP.__init__(self, nb_feature, nb_hidden, nb_class) 
        self.alphalr = alphalr
 
    def fit(self, X, Y, nb_epoch=1):
        for _ in range(nb_epoch):
            # Store current parameters
            curr_W1, curr_b1, curr_W2, curr_b2 = self.dense1.W, self.dense1.b, self.dense2.W, self.dense2.b           
            # Store current parameter loss
            curr_loss = self.forward(X, Y) 
 
            dL = np.ones([len(X), 1], dtype='float64')
            dS = self.softmaxce.backward(dL)    
            dH, dW2, db2 = self.dense2.backward(dS)
            dZ = self.sigmoid.backward(dH)
            dX, dW1, db1 = self.dense1.backward(dZ)
            
            # New parameters after GD
            self.dense1.W = self.dense1.W - (self.alphalr * dW1)
            self.dense1.b = self.dense1.b - (self.alphalr * db1)
            self.dense2.W = self.dense2.W - (self.alphalr * dW2)
            self.dense2.b = self.dense2.b - (self.alphalr * db2)
            
            new_loss = self.forward(X, Y)

            # If new parameters give higher loss, revert to old parameters 
            if curr_loss < new_loss:                               
                self.dense1.W, self.dense1.b, self.dense2.W, self.dense2.b = curr_W1, curr_b1, curr_W2, curr_b2  
    
    def _get_gradients(self, X, Y):
        curr_loss = self.forward(X, Y) 
        dL = np.ones([len(X), 1], dtype='float64')
        dS = self.softmaxce.backward(dL)    
        dH, dW2, db2 = self.dense2.backward(dS)
        dZ = self.sigmoid.backward(dH)
        dX, dW1, db1 = self.dense1.backward(dZ)
        return dX, dW1, db1, dZ, dH, dW2, db2, dS                

import tensorflow as tf

class TFMLP():    
    def __init__(self, nb_feature, nb_hidden, nb_class):
        self.X = tf.placeholder(tf.float32, [None, nb_feature], name = 'X')
        self.Y = tf.placeholder(tf.float32, [None, nb_class], name = 'Y')
        self.W1 = tf.Variable(initial_value=np.random.randn(nb_feature, nb_hidden), dtype=tf.float32, name = 'W1')
        self.b1 = tf.Variable(initial_value=np.random.randn(nb_hidden), dtype=tf.float32, name = 'b1')        
        self.W2 = tf.Variable(initial_value=np.random.randn(nb_hidden, nb_class), dtype=tf.float32, name = 'W2')
        self.b2 = tf.Variable(initial_value=np.random.randn(nb_class), dtype=tf.float32, name = 'b2')
        
        self._f()
        
    def _f(self):
        Z = tf.add(tf.matmul(self.X, self.W1), self.b1)
        H = tf.div(tf.constant(1.0),
                    tf.add(tf.constant(1.0), tf.exp(tf.negative(Z))))
        S = tf.add(tf.matmul(H, self.W2), self.b2)  
        self.P = tf.nn.softmax(S)        
        correct_prediction = tf.equal(tf.argmax(self.Y,axis=1), tf.argmax(self.P,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.P), reduction_indices=[1]))        
       
    def predict(self, X):       
        return self.P.eval(feed_dict={self.X:X})

    def evaluate(self, X, Y):  
        self.P.eval(feed_dict={self.X:X})
        return self.accuracy.eval(feed_dict={self.X:X, self.Y:Y})
    
    def forward(self, X, Y):
        return self.loss.eval(feed_dict={self.X:X, self.Y:Y})
        
class TFMLPWithGDOptimizer(TFMLP):    
    def __init__(self, nb_feature, nb_hidden, nb_class):
        TFMLP.__init__(self, nb_feature, nb_hidden, nb_class)
       
    def fit(self, X, Y, sess, nb_epoch=10, learningrate=0.5):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learningrate).minimize(self.loss)
        for _ in range(nb_epoch):          
            sess.run(fetches=[train_step, self.loss, self.accuracy], feed_dict={self.X: X, self.Y: Y})    