import matplotlib.pyplot as plt 

import queue
import sys 
sys.setrecursionlimit(10**6)
import numpy  as np 

def euclidean_distance(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    dis = (p1 - p2)**2
    return np.sqrt(np.sum(dis))


    
   
class Node(object):
    def __init__(self,number, location, spectrum, max_lim_x, max_lim_y,master = None, channel = 0):
        self.number = number
        self.master = master
        self.queue  = queue.Queue() # Holding the message
        self.location_xy =  np.asarray(location, dtype = np.float32)
        self.spectrum = spectrum
        self.max_lim_x = max_lim_x
        self.max_lim_y = max_lim_y
        self.channel = channel
        self.number_of_channels = self.spectrum.action_space
        
    def get_message(self):
        if self.queue.empty():
            return None 
        else:
            return self.queue.get()
    
    def insert_message(self, m):
        self.queue.put(m)
        
    def move(self, del_x = None, del_y= None, std = 0.1):
        if del_x is None and del_y is None:
            del_x , del_y = np.random.normal(loc=0.0 , scale = std, size = 2)
        elif del_x is None and del_y is not None:
            del_x = np.ranodm.normal(loc = 0.0, scale = std, size = 1)
        elif del_y is None and del_x is not None:
            del_y = np.ranodm.normal(loc = 0.0, scale = std, size = 1)


        self.location_xy += np.asarray([del_x, del_y])
        
        if self.location_xy[0] > self.max_lim_x:
            self.location_xy[0] = self.max_lim_x
        elif self.location_xy[0] < -self.max_lim_x:
            self.location_xy[0] = -self.max_lim_x
            
        if self.location_xy[1] > self.max_lim_y:
            self.location_xy[1] = self.max_lim_y
        elif self.location_xy[1] < -self.max_lim_y:
            self.location_xy[1] = -self.max_lim_y
        
        
    def sense_spectrum(self, channel, how_many = 0):
        a = channel 
        b = channel + how_many 
        s = self.spectrum.sense([a,b])
        vec = np.asarray([-1]*self.number_of_channels)
        vec[a:b+1] = s
        return vec
        
class Graph(object):
    
    def __init__(self, all_nodes, topology = None):
        # Set the connection matrix
        self.nVertices = len(all_nodes)
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
        self.connectionMatrix = np.zeros(shape = (self.nVertices, self.nVertices), dtype = np.int32)
        self.all_nodes = all_nodes
        self.topology = topology
        
        self.update_adjMatrix()
        self.update_connectionMatrix()
   
    def moveNodes(self):
        for n in self.all_nodes:
            n.move()
            
        self.update_adjMatrix()
        self.update_connectionMatrix()
        
    def update_adjMatrix(self):
        
        for i in range(self.nVertices):
            n1 = self.all_nodes[i]
            for j in range(i+1, self.nVertices):
                n2 = self.all_nodes[j]
                self.addEdge(n1, n2)
            
    def update_connectionMatrix(self):
        for i in range(self.nVertices):
            node_i = self.all_nodes[i]
            for j in range(i+1, self.nVertices):
                node_j = self.all_nodes[j]
                if (self.adjMatrix[i][j] < 1 and node_i.channel == node_j.channel and 
                node_i.spectrum.current_spectrum[node_i.channel] == 0 and node_j.spectrum.current_spectrum[node_j.channel] == 0) :
                    val  = 1
                else:
                    val  = 0
                
                self.connectionMatrix[i][j] = val
                self.connectionMatrix[j][i] = val
                    
        
        
    def addEdge(self, node1, node2):
        v1 = node1.number
        v2 = node2.number
        
        pointV1 = self.topology.get_location(node1.location_xy)
        pointV2 = self.topology.get_location(node2.location_xy)
        dis = euclidean_distance(pointV1, pointV2)
        
        self.adjMatrix[v1][v2] = dis
        self.adjMatrix[v2][v1] = dis
        
    

