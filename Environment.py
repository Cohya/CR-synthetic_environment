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



def generateRandoNode(num,num_of_channels ,max_lim_x, max_lim_y, master):
    min_lim_x = - max_lim_x
    min_lim_y = - max_lim_y
    x, y = [(max_lim_x-min_lim_x), (max_lim_y - min_lim_y)] * np.random.random(size = 2) + [min_lim_x, min_lim_y]
    spectrum  = SingleSpectrum(number_of_channels=num_of_channels)
    node = Node(num, location = [x,y], spectrum=spectrum, max_lim_x = max_lim_x, max_lim_y = max_lim_y, 
                master=master)
    return node

def generate_list_of_nodes(num_of_nodes ,num_of_channels , max_lim_x, max_lim_y):
    all_nodes = []
    for i in range(num_of_nodes):
        if i == 0 :
            master = True 
        else:
            master = False
            
        all_nodes.append(generateRandoNode(i, num_of_channels ,max_lim_x, max_lim_y, master = master))
    return all_nodes


class SimulatedEnv(object):
    def __init__(self, jammers, num_of_channels = 5, num_of_nodes = 5, topology = Topology(func = func), deliveryRate =0):
        self.topology = topology
        self.num_of_nodes = num_of_nodes
        self.max_lim_x = 5
        self.max_lim_y = 5
        self.all_nodes = generate_list_of_nodes(num_of_nodes,num_of_channels,0.5,0.5)
        self.graph = Graph(all_nodes = self.all_nodes, topology= self.topology, deliveryRate = deliveryRate )
    
        self.whoSendMessage = 0 # round
        self.num_of_channels = num_of_channels
        self.jammers = jammers
        
        self.info = {'Real_Spectrum': None, 
                     'sensed_spectrum': None,
                     'last_reward': None,
                     'number_of_jammers:': len(self.jammers), 
                     'number_of_channels': self.num_of_channels}
        
        self.full_mesh_connection = self.num_of_nodes*(self.num_of_nodes-1)/2
        
    def get_current_spectrum(self):
        spectrum = []
        for node in self.all_nodes:
            spectrum.append(node.spectrum.current_spectrum)
        return spectrum
    
    def close_image(self):
        plt.close('all')
        self.image_active = False
        
    def imageOfGame(self):
        self.graph.update_connectionMatrix()
        if self.image_active == False:
            X, Y = np.meshgrid(np.linspace(-self.max_lim_x, self.max_lim_x, 256), np.linspace(-self.max_lim_y, self.max_lim_y, 256))
            Z = self.topology.func(X,Y)
            self.levels = np.linspace(Z.min(), Z.max(), 7)
            self.figure, self.ax = plt.subplots(1)
            contourf_ = self.ax.contourf(X, Y, Z, levels=self.levels)
            plt.title("Based xy euclidian distance")
            self.figure.colorbar(contourf_)
            self.d = {}
            for i in range(self.num_of_nodes):
                for j in range(i+1, self.num_of_nodes):
                    n1 = self.all_nodes[i]
                    if self.graph.connectionMatrix[i][j] > 0:
                        n2 = self.all_nodes[j]
                        x = [n1.location_xy[0], n2.location_xy[0]]
                        y = [n1.location_xy[1], n2.location_xy[1]]
                        if self.graph.connectionMatrix[i][j] == 1:
                            co_i = 'k'
                        else:
                            co_i = 'b'
                            
                        line , = self.ax.plot(x,y, color = co_i)
                        self.d[(i,j)] = line
                    else:
                        line , = self.ax.plot(0,0, 'k' )
                        self.d[(i,j)] = line
            
            self.d_jammers = {}
            count = 0
            for j in self.jammers:
                s1 = self.ax.scatter(j.location_xy[0],j.location_xy[1], color = 'r' ,s = 60, marker = 'o',facecolors="None" ,linewidth=2)
                s2 = self.ax.scatter(j.location_xy[0],j.location_xy[1], color = 'r' ,s = 70, marker = 'x')
                circle = plt.Circle((j.location_xy[0],j.location_xy[1]), j.radius, color='r',  alpha=0.1)
                circle_line = self.ax.add_patch(circle)
                self.d_jammers[(count, 'l1')] = s1
                self.d_jammers[(count, 'l2')] = s2
                self.d_jammers[(count, 'c1')] = circle
                self.d_jammers[(count, 'cl')] = circle_line 
                count += 1
                
            self.image_active = True
            x, y = get_all_x_y(self.all_nodes)
            self.s = self.ax.scatter(x, y, c = 'r')
        else:
            #figure.canvas.flush_events()
            
            for i in range(self.num_of_nodes):
                n1 = self.all_nodes[i]
                for j in range(i+1, self.num_of_nodes):
                    
                    if self.graph.connectionMatrix[i][j] > 0:
                        n2 = self.all_nodes[j]
                        x = [n1.location_xy[0], n2.location_xy[0]]
                        y = [n1.location_xy[1], n2.location_xy[1]]
                        self.d[(i,j)].set_xdata(x)
                        self.d[(i,j)].set_ydata(y)
                        if self.graph.connectionMatrix[i][j] == 1:
                            self.d[(i,j)].set_color('k')
                        else:
                        
                            self.d[(i,j)].set_color('b')
                    else:
                        self.d[(i,j)].set_xdata(0)
                        self.d[(i,j)].set_ydata(0)
                        
                count = 0
                for j in self.jammers:
                    
                    x,y = j.location_xy[0],j.location_xy[1]
                    
                    s1 = self.ax.scatter(x,y, color = 'r' ,s = 60, marker = 'o',facecolors="None" ,linewidth=2)
                    s2 = self.ax.scatter(x,y, color = 'r' ,s = 70, marker = 'x')
                    self.d_jammers[(count, 'l1')].remove() 
                    self.d_jammers[(count, 'l2')].remove() #set_array([x,y])
                    self.d_jammers[(count, 'l1')] = s1
                    self.d_jammers[(count, 'l2')] = s2

                    self.d_jammers[(count, 'c1')].center = (x,y)

                    count += 1
            
            x1, y1 = get_all_x_y(self.all_nodes)
            self.s.remove()
            self.s = self.ax.scatter(x1, y1, c = 'r')
            
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            
            # for j in self.jammers:
            #     j.move()
            # for n in self.all_nodes:
            #     n.move()
            
    def move_jammers(self):
        for j in self.jammers:
            j.move()
            
    def move_nodes(self):
        self.graph.moveNodes()
        
        
    def euclidian_distance(self, node,jammer):
        node_xyz = self.topology.get_location(node.location_xy)
        jammer_xyz = self.topology.get_location(jammer.location_xy)
        
        dis = (node_xyz - jammer_xyz)**2
        return np.sqrt(np.sum(dis)) 

    def euclidian_distance_xy(self, node,jammer):
        node_xy= node.location_xy
        jammer_xy= jammer.location_xy
        
        dis = (node_xy - jammer_xy)**2
        return np.sqrt(np.sum(dis)) 
   
    def run_jammers(self):
        for jammer in self.jammers:
            action = jammer.get_action()
            
            for node in self.all_nodes:
                # dis = self.euclidian_distance(node,jammer) # for xyz distance in 3D
                dis = self.euclidian_distance_xy(node,jammer)
                if dis < jammer.radius:
                    node.spectrum.step(action, noise= True)
    
    def clear_spectrum(self):
        for node in self.all_nodes:
            node.spectrum.current_spectrum = [0]*self.num_of_channels
            
    def sense_spectrum(self):
        state = []
        for node in self.all_nodes:
            if self.graph.master_node.channel == node.channel:
                channel_to_sense = np.random.choice(self.num_of_channels)
                s = node.sense_spectrum(channel_to_sense)
            else:
                s = [None] * self.num_of_channels
            
            state.append(s)
            
        return np.asarray(state)#, dtype = np.int32)
        
        
        
    def step(self, action):
       # send all channel info from all nodes to the master 
       
        # distribute the master message to all graph 
       self.graph.distribute_master_decision(action) 

       self.clear_spectrum()
       self.run_jammers()
       self.graph.update_connectionMatrix()
       # c = self.communication_channel 
       print("connection matrix", self.graph.connectionMatrix)
       r = np.sum(self.graph.connectionMatrix)/(2*self.full_mesh_connection) #give a measure of connection  as radio of full mesh (in full mesh we have n-1 + n-2 + ...1 = (n-1)*(n)/2)
       observation = self.sense_spectrum()
       self.time_counter += 1
       
       self.info['Real_Spectrum'] = self.get_current_spectrum()
       self.info['sensed_spectrum'] = observation
       self.info['last_reward'] = r
       
       if self.time_counter >= 100:
           done = True 
       else:
           done = False
           
       return observation, r, done, self.info  
       # print(next_state)
       
       
       
    
    def reset_jammers(self):
        for j in self.jammers:
            j.reset()
      
    
    def reset(self):
        self.reset_jammers()
        self.image_active = False
        self.time_counter = 0
        
        for node in self.all_nodes:
            # we build the net on channel 0 
            # in the future we can create a net builder function with somew search 
            node.channel = 0
            
        self.graph.update_connectionMatrix()
        self.communication_channel = 0
            

    
    
        













