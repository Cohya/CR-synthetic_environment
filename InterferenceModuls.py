import numpy as np 


class Jammer(object):
    def __init__(self,location_xy, r, frequency, num_of_channels, starting_channel = None):
        """
        location_xy - an array of x y corrdinate 
        r - is the radius of effect 
        frequency - the velocity of changing channels (if frequency == 0 then it is a 'spot Jamming')
        num_of_channels - number of possible channels the JAmmer can interfere
        starting_channel - the initial channel  
        """
        self.location_xy = np.asarray(location_xy, dtype = np.float32)
        if frequency == 0:
            self.typee = 'Spot_Jamming'
        else:
            self.typee = 'Sweep_jamming'
            
        self.frequency = frequency
        self.radius = r
        self.num_of_channels = num_of_channels
        self.starting_channel = starting_channel

    def reset(self):
        self.time = 0
        if self.starting_channel is None:
            self.current_channel  = np.random.choice(self.num_of_channels)
        else:
            self.current_channel = self.starting_channel

        
    def get_action(self):
        # returnthe channel it interfere
        self.time += 1
        if self.time % self.frequency == 0:
            self.current_channel = (self.current_channel+1)%self.num_of_channels
        
        return self.current_channel
    
    def move(self, del_x = None, del_y = None, lim_x = 5 , lim_y = 5):

        if (del_x is None) and (del_y is None):
            del_x, del_y = np.random.randn(2) * np.sqrt(0.1) ## it will be ~N(0,0.1)
        elif del_x is None or del_y is None:
            val  = np.random.randn(1) * np.sqrt(0.1)
            if del_x is None:
                del_x = val 
            else:
                del_y = val 
                
        self.location_xy[0] += del_x
        self.location_xy[1] += del_y
        
        if np.abs(self.location_xy[0]) > lim_x:
            if self.location_xy[0] > 0:
                self.location_xy[0] = lim_x
            else:
                self.location_xy[0] = - lim_x
                
        if np.abs(self.location_xy[1]) > lim_y:
            if self.location_xy[1] > 0:
                self.location_xy[1] = lim_y
            else:
                self.location_xy[1] = - lim_y
        
        
    
            