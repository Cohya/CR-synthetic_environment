from Environment import SimulatedEnv
from InterferenceModuls import Jammer
import matplotlib.pyplot as plt 
import numpy as np

plt.close('all')
jammer_1 = Jammer(location_xy = [1,1], r = 3, frequency = 1, num_of_channels = 5)
jammer_2 = Jammer(location_xy = [0.5,-1], r = 3, frequency = 1, num_of_channels = 5)
num_of_channels = 5
env = SimulatedEnv( num_of_channels =num_of_channels, num_of_nodes=5,jammers = [jammer_1, jammer_2], deliveryRate = 4)
env.reset()
print(env.graph.adjMatrix)
import time 
for i in range(1000):
    env.graph.all_nodes[-1].channel = np.random.choice(num_of_channels)
    obs, r,_,_ = env.step(0)
    
    print("reward:", r)
    print("Observation:" ,obs)
    for j in env.jammers:
        print(j.current_channel)
       
    for node in env.graph.all_nodes:
        print(node.channel, end = " ")
    print("")
    print(np.asarray(env.get_current_spectrum()))
    
    print("-------------------------------------")
    # print(env.communication_channel)
    time.sleep((0.002))
    env.imageOfGame()
    env.move_nodes() #change position og nodes randomly 
    env.move_jammers() # change position of jammers randomly
    