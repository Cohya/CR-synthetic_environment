from Environment import SimulatedEnv
from InterferenceModuls import Jammer
import matplotlib.pyplot as plt 
import numpy as np

plt.close('all')
jammer_1 = Jammer(location_xy = [1,1], r = 3, frequency = 1, num_of_channels = 5)
jammer_2 = Jammer(location_xy = [0.5,-1], r = 3, frequency = 1, num_of_channels = 5)

env = SimulatedEnv(num_of_nodes=10,jammers = [jammer_1, jammer_2])
env.reset()

import time 
for i in range(10):
    
    obs, r,_,_ = env.step(0)
    print("reward:", r)
    print("Observation:" ,obs)
    for j in env.jammers:
        print(j.current_channel)
        
    print(np.asarray(env.get_current_spectrum()))
    # print(env.communication_channel)
    time.sleep((1))
    env.imageOfGame()
# env.imageOfGame()
# env.step()
# 

# for j in env.jammers:
#     print(j.current_channel)

# print(env.get_current_spectrum())
# env.step(1)
# print(env.get_current_spectrum())