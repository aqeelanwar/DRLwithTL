
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    ###################################################################

    def get_weights(self):
        pass
        
    def num_params(self):
        pass

    def forward(self, X):
        pass

    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        pass

    def gv(self, AI, AO, DO):    
        pass
        
    def train(self, AI, AO, DO): 
        pass
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        pass
        
    def dfa_gv(self, AI, AO, E, DO):
        pass
        
    def dfa(self, AI, AO, E, DO): 
        pass
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel_gv(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
