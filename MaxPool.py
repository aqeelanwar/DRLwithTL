
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops
# return gen_nn_ops.max_pool_v2(value=X, ksize=self.size, strides=self.strides, padding="SAME")

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class MaxPool(Layer):
    def __init__(self, size, ksize, strides, padding):
        self.size = size
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    def forward(self, X):
        Z = tf.nn.max_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # Z = tf.Print(Z, [Z], message="", summarize=1000)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
