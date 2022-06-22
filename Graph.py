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

def findminIndex(vec, visited):
    min_val = sys.maxsize
    index = None
    for i in range(len(vec)):
        val = vec[i]
        if visited[i] == True:
            continue 
        
        if val <= min_val:
            min_val = val
            index = i
            
    return index ,val 
    
   
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
    
    def __init__(self, all_nodes, topology = None, deliveryRate = 0):
        # Set the connection matrix
        self.nVertices = len(all_nodes)
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
        self.connectionMatrix = np.zeros(shape = (self.nVertices, self.nVertices), dtype = np.float32)
        # self.messageConnectionMatrix = np.zeros(shape = (self.nVertices, self.nVertices), dtype = np.int32)
        self.all_nodes = all_nodes
        self.topology = topology
        self.deliveryRate = deliveryRate
        
        self.update_adjMatrix()
        self.update_connectionMatrix()
        # self.update_messageConnectionMatrix()
        
        
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
                
                #check if they both on the same channel (destination and source) + if the channel is free for both 
                if (node_i.channel == node_j.channel and node_i.spectrum.current_spectrum[node_i.channel] == 0 and node_j.spectrum.current_spectrum[node_j.channel] == 0):
                    # check the distance between them (goodness of direct connection)
                    if (self.adjMatrix[i][j] < 1 ) :# we can change 1 with some other param value
                        val  = 1#self.adjMatrix[i][j] # 1 or the distance 
                     
                    # if there is no direct connection, check if you can reach the node through delivery up to delivery rate 
                    elif self.deliveryRate > 0:
                        # a series of node you should use in order to reach the destination node_i->node_j, you'll get direct_path= [node_j, ..., node_i]
                        distance,_,direct_path = self.findShortestPaths(node_i , node_j)
                        # print(distance)
                        # print("DP:",direct_path)
                        if  direct_path is not None and len(direct_path) <= self.deliveryRate + 1:
                            # in case there is a path -> check if all nodes in the same channel 
                            val = 0.5
                            # print("In")
                            for n in direct_path:
                                if self.all_nodes[n].channel != node_i.channel:
                                    val  = 0
                                    # print("Bomm")
                                    break
                        else:
                            val = 0 
                    else:
                        val = 0
                                    
                else:
                    val  = 0
                
                # print("v",val, i,j)
                self.connectionMatrix[i][j] = val
                self.connectionMatrix[j][i] = val
                    
    
    
    def findShortestPaths(self, node1, node2):
        """ (Dijkstras Algo)""" 
        v1 = node1.number
        v2 = node2.number
        visited = [False] * self.nVertices 
        distance = [sys.maxsize] * self.nVertices 
        distance[v1] = 0
        path  = [None] * self.nVertices 
        for i in range(self.nVertices):
             # choose vertex with min distance
            index,_ = findminIndex(distance, visited) # O(n)
            visited[index] = True
            for j in range(self.nVertices):
                if visited[j] == False:
                    val = distance[index] + self.adjMatrix[index][j] 
                    # print(val)
                    if  distance[j] > val:
                        distance[j] = val 
                        path[j] = index
        
        direct_path = []
        if path[v2] is None:
            return distance[v2], path, None
        
        direct_path.append(v2)
        v_d = v2
        while v_d != v1:
            v_d = path[v_d]
            direct_path.append(v_d)
            
            
        return distance[v2], path , direct_path
    
    # def update_messageConnectionMatrix(self):
        
    #     for i in range(self.nVertices):
    #         node_i = self.all_nodes[i]
    #         for j in range(i+1, self.nVertices):
    #             node_j = self.all_nodes[j]
    #             if (self.adjMatrix[i][j] < 1 and node_i.channel == node_j.channel and 
    #             node_i.spectrum.current_spectrum[node_i.channel] == 0 and node_j.spectrum.current_spectrum[node_j.channel] == 0) :
    #                 val  = 1#self.adjMatrix[i][j] # 1 or the distance 
                
    #             elif self.findShortestPaths(node_i , node_j)[2] is not None and len(self.findShortestPaths(node_i , node_j)[2]) <= self.deliveryRate + 1:
    #                 val  = 1
    #             else:
    #                 val  = 0
                
    #             self.messageConnectionMatrix[i][j] = val
    #             self.messageConnectionMatrix[j][i] = val
                    
        
    def addEdge(self, node1, node2):
        v1 = node1.number
        v2 = node2.number
        
        pointV1 = self.topology.get_location(node1.location_xy)
        pointV2 = self.topology.get_location(node2.location_xy)
        dis = euclidean_distance(pointV1, pointV2)
        
        if dis >= 1:
            dis = sys.maxsize
            
        self.adjMatrix[v1][v2] = dis
        self.adjMatrix[v2][v1] = dis
        
    

