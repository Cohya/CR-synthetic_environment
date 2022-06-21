import numpy as np 

class SingleSpectrum(object):
    """Can ALso be called shared spectrum"""
    
    def __init__(self, number_of_channels = 5):

        self.action_space = number_of_channels
        self.current_spectrum = [0] * self.action_space
        self.info = {"spectrum" : self.current_spectrum}
   
    
        
    def step(self,action):
        # action is a touple(i mod N)
        # i --> sending massege 
        # j --> sense

        if self.current_spectrom[action] == 1 : # means there is a PU there or a noise 
            r = -1
            
        else:  
            # self.current_spectrom[action] = 1 ## means the channel is free
            r = 1
           
        return r 
       
    def clean(self):
         self.current_spectrum = [0] * self.action_space
        
    def sense(self, interval):
        # interval == [a,b]
        a = int(interval[0])
        b = int(interval[1]+1)
        
        return self.current_spectrom[a:b]