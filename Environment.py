from Graph import Graph, Node
import numpy as np 
from Spectrum import SingleSpectrum
import matplotlib.pyplot as plt 

class Topology(object):
    def __init__(self, func = None):
        self.func = func
        
    def get_location(self, point):
        x, y  = point 
        z = self.func(x, y)
        
        return np.array([x, y, z])
    
def func(X, Y):
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    return Z


def get_all_x_y(all_nodes):
    x = []
    y = []
    for n in all_nodes:
        x.append(n.location_xy[0])
        y.append(n.location_xy[1])
    return x, y 


def generateRandoNode(num,max_lim_x, max_lim_y):
    min_lim_x = - max_lim_x
    min_lim_y = - max_lim_y
    x, y = [(max_lim_x-min_lim_x), (max_lim_y - min_lim_y)] * np.random.random(size = 2) + [min_lim_x, min_lim_y]
    spectrum  = SingleSpectrum()
    node = Node(num, location = [x,y], spectrum=spectrum, max_lim_x = max_lim_x, max_lim_y = max_lim_y)
    return node

def generate_list_of_nodes(num_of_nodes , max_lim_x, max_lim_y):
    all_nodes = []
    for i in range(num_of_nodes):
        all_nodes.append(generateRandoNode(i, max_lim_x, max_lim_y))
    return all_nodes


class SimulatedEnv(object):
    def __init__(self, num_of_nodes = 5, topology = Topology(func = func)):
        self.topology = topology
        self.num_of_nodes = num_of_nodes
        self.max_lim_x = 5
        self.max_lim_y = 5
        self.all_nodes = generate_list_of_nodes(num_of_nodes, 0.5,0.5)
        self.graph = Graph(all_nodes = self.all_nodes, topology= self.topology)
    
        self.whoSendMessage = 0 # round
        self.image_active = False
        # self.num_of_channels = 
  
    def close_image(self):
        plt.close('all')
        self.image_active = False
        
    def imageOfGame(self):
        if self.image_active == False:
            X, Y = np.meshgrid(np.linspace(-self.max_lim_x, self.max_lim_x, 256), np.linspace(-self.max_lim_y, self.max_lim_y, 256))
            Z = self.topology.func(X,Y)
            self.levels = np.linspace(Z.min(), Z.max(), 7)
            self.figure, self.ax = plt.subplots(1)
            contourf_ = self.ax.contourf(X, Y, Z, levels=self.levels)
            self.figure.colorbar(contourf_)
            self.d = {}
            for i in range(self.num_of_nodes):
                for j in range(i+1, self.num_of_nodes):
                    n1 = self.all_nodes[i]
                    if self.graph.connectionMatrix[i][j] == 1:
                        n2 = self.all_nodes[j]
                        x = [n1.location_xy[0], n2.location_xy[0]]
                        y = [n1.location_xy[1], n2.location_xy[1]]
                        line , = self.ax.plot(x,y, 'k' )
                        self.d[(i,j)] = line
                    else:
                        line , = self.ax.plot(0,0, 'k' )
                        self.d[(i,j)] = line
                        
            self.image_active = True
            x, y = get_all_x_y(self.all_nodes)
            self.s = self.ax.scatter(x, y, c = 'r')
        else:
            #figure.canvas.flush_events()
            
            for i in range(self.num_of_nodes):
                n1 = self.all_nodes[i]
                for j in range(i+1, self.num_of_nodes):
                    
                    if self.graph.connectionMatrix[i][j] == 1:
                        n2 = self.all_nodes[j]
                        x = [n1.location_xy[0], n2.location_xy[0]]
                        y = [n1.location_xy[1], n2.location_xy[1]]
                        self.d[(i,j)].set_xdata(x)
                        self.d[(i,j)].set_ydata(y)
                    else:
                        self.d[(i,j)].set_xdata(0)
                        self.d[(i,j)].set_ydata(0)
            
            x1, y1 = get_all_x_y(self.all_nodes)
            self.s.remove()
            self.s = self.ax.scatter(x1, y1, c = 'r')
        
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.graph.moveNodes()
    def step(self):
       pass
   
    
    def init(self):
        self.image_active = False
        pass

    
    
        
















# topology = Topology(func = func)
# node1 = Node(0, location = [1,1])
# node2 = Node(1, location= [2,2])
# node3 = Node(2, location= [1.2,2])
# all_nodes = [node1, node2, node3]
# g = Graph(len(all_nodes),all_nodes , topology = topology)


# X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
# Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# levels = np.linspace(Z.min(), Z.max(), 7)

# # plot
# figure, ax = plt.subplots(1)

# contourf_ = ax.contourf(X, Y, Z, levels=levels)
# figure.colorbar(contourf_)
# d = {}
# for i in range(3):
#     for j in range(i+1, 3):
#         n1 = all_nodes[i]
#         if g.connectionMatrix[i][j] == 1:
#             n2 = all_nodes[j]
#             x = [n1.location_xy[0], n2.location_xy[0]]
#             y = [n1.location_xy[1], n2.location_xy[1]]
#             line , = ax.plot(x,y, 'k' )
#             d[(i,j)] = line
#         else:
#             line , = ax.plot(0,0, 'k' )
#             d[(i,j)] = line
            
# print(d)
# x, y = get_all_x_y(all_nodes)
# s = ax.scatter(x, y, c = 'r')

# import time
# for i in range(50):
    
#     #figure.canvas.flush_events()
#     for i in range(3):
#         for j in range(i+1, 3):
#             n1 = all_nodes[i]
#             if g.connectionMatrix[i][j] == 1:
#                 n2 = all_nodes[j]
#                 x = [n1.location_xy[0], n2.location_xy[0]]
#                 y = [n1.location_xy[1], n2.location_xy[1]]
#                 d[(i,j)].set_xdata(x)
#                 d[(i,j)].set_ydata(y)
#             else:
#                 d[(i,j)].set_xdata(0)
#                 d[(i,j)].set_ydata(0)
    
#     x1, y1 = get_all_x_y(all_nodes)
#     s.remove()
#     s = ax.scatter(x1, y1, c = 'r')

#     figure.canvas.draw()
#     figure.canvas.flush_events()
#     g.moveNodes()
#     time.sleep(1)
  