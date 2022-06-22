import sys 

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
            
    
class Graph(object):
    def __init__(self,n):
        self.adjMatrix = [[sys.maxsize]*n for _ in range(n)]
        self.n = n
    
    def addEdge(self, v1, v2, value):
        self.adjMatrix[v1][v2] = value
        self.adjMatrix[v2][v1] = value
        
    
    def shortestPath(self, v1, v2):
        visited = [False] * self.n
        distance = [sys.maxsize] * self.n
        distance[v1] = 0
        path  = [None] * self.n
        for i in range(self.n):
            # choose vertex with min distance

            index,_ = findminIndex(distance, visited) # O(n)
            print("index:", index)
            visited[index] = True
            print(distance)
            for j in range(self.n):
                if visited[j] == False:
                    val = distance[index] + self.adjMatrix[index][j] 
                    # print(val)
                    if  distance[j] > val:
                       distance[j] = val 
                       path[j] = index
        
        direct_path= []
        if path[v2] is None:
            return distance[v2], path, None
        
        direct_path.append(v2)
        v_d = v2
        while v_d != v1:
            v_d = path[v_d]
            direct_path.append(v_d)
            
            
        return distance[v2], path , direct_path
                
            
            

g = Graph(5)
g.addEdge(0,1,6)
g.addEdge(0,3,1)
g.addEdge(1,2,5)
g.addEdge(1,3,2)
g.addEdge(1,4,2)
g.addEdge(2,4, 5)
g.addEdge(3, 4,1)
print(g.adjMatrix)

print(g.shortestPath(0, 2))

