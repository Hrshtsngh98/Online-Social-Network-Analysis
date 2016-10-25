from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
   g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    visited = set()
    q = [root]
    d = 0
    node2distances = {}
    node2num_paths = {}
    node2parents = {}
    node2distances[root] = 0
    node2num_paths[root] = 1

    while len(q)>0:
      v = q.pop(0)
      num_of_paths = node2num_paths[v]
      dist_root = node2distances[v]
      if node2distances[v] < max_depth:
        for vertex in graph[v]:
            if vertex not in node2distances:
                node2distances[vertex] = dist_root + 1
                q.append(vertex)
            if node2distances[vertex] == dist_root + 1:
                if vertex in node2num_paths:
                    node2num_paths[vertex] = node2num_paths[vertex] + num_of_paths
                else:
                    node2num_paths[vertex]=num_of_paths

                if vertex in node2parents:
                    node2parents[vertex].append(v)
                else:
                    node2parents[vertex]=[v]

    return node2distances, node2num_paths, node2parents


def complexity_of_bfs(V, E, K):
 return (V+E)


def bottom_up(root, node2distances, node2num_paths, node2parents):
    node2parents[root]=[]
    allnodes = []
    d= {}
    result = {}
    new_distances = []
    
    for dist in node2distances:
       new_distances.append((dist,node2distances[dist]))
    new_distances=sorted(new_distances,key=lambda x:(-x[1],x[0]))

    for dist in new_distances:
        allnodes.append(dist[0])
   
    for dist in node2distances:
       d[dist]=1

    new_distances=sorted(new_distances,key=lambda x:(-x[1],x[0])) 
    for node in allnodes:
      for parent in node2parents[node]:
        d[parent] = d[parent]+(d[node]/len(node2parents[node]))
        if parent>node:
            result[(node,parent)] = d[node]/len(node2parents[node])
        else:
            result[parent,node] = d[node]/len(node2parents[node])

    return result


def approximate_betweenness(graph, max_depth):
   betweenness ={}
    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph,node,max_depth)
        result =bottom_up(node, node2distances, node2num_paths, node2parents)
        for edge in result:
            if edge in betweenness:
                betweenness[edge] = betweenness[edge] +result[edge]
            else:
                betweenness[edge] = result[edge]

    for edge in betweenness:
        betweenness[edge]=betweenness[edge]/2
    
    return betweenness


def is_approximation_always_right():
    return 'no'


def partition_girvan_newman(graph, max_depth):
    result = []
    components =[]
    edges =graph.edges()
    copy = graph.copy()
    betweenness = approximate_betweenness(graph, max_depth)
    edges_to_delete = sorted(betweenness.items(), key=lambda x:(x[1],x[0][0],x[0][1]), reverse=True)
    #print(edges_to_delete)
    count = 0
    while nx.number_connected_components(copy)<2:
      for edge in edges_to_delete:
        if (edge[0][0],edge[0][1] in copy):
            copy.remove_edge(edge[0][0],edge[0][1])
        elif (edge[0][1],edge[0][0] in copy):
            copy.remove_edge(edge[0][1],edge[0][0])
        if nx.number_connected_components(copy)>1:
            components =[c for c in nx.connected_component_subgraphs(copy)]
            #print(len(components[0]),len(components[1]))
            return components

def get_subgraph(graph, min_degree):
    n = []
    for node in graph.nodes():
        if len(graph.neighbors(node))<min_degree:
            n.append(node)
    graph.remove_nodes_from(n)
    return graph

def volume(nodes, graph):
    count=0

    edges_in_graph = graph.edges()
    edge_set = set()
    for n in nodes:
      for edge in edges_in_graph:
        if (edge[0]==n or edge[1]==n) and ((edge[0],edge[1]) not in edge_set or (edge[0],edge[1]) not in edge_set):
            count=count+1
            edge_set.add(edge)

    return count


def cut(S, T, graph):
    count = 0
    for y in graph.edges():
        if (y[0] in S and y[1] in T) or (y[0] in T and y[1] in S):
            count = count + 1
    return count 


def norm_cut(S, T, graph):
    cutvalue = (cut(S,T,graph))
    volumeS = (volume(S,graph))
    volumeT = (volume(T,graph))
    
    value = (cutvalue/volumeS) + (cutvalue/volumeT)
    return float(value)


def score_max_depths(graph, max_depths):
    result = []
    for depth in max_depths:
        c=partition_girvan_newman(graph, depth)
        normcut = norm_cut(c[0].nodes(),c[1].nodes(), graph)
        result.append((depth, normcut))
    return result


def make_training_graph(graph, test_node, n):
   copy = graph.copy()
    edgetoremove = []
    neigh=copy.neighbors(test_node)
    neigh=sorted(neigh)
    for i in range(0,n):
        edgetoremove.append((neigh[i],test_node))
    copy.remove_edges_from(edgetoremove)
    return copy



def jaccard(graph, node, k):
    scores = []
    neighborsi = set(graph.neighbors(node))
    for n in graph.nodes():
        if n != node and not graph.has_edge(node, n) and not graph.has_edge(node,n):
            neighborsj = set(graph.neighbors(n))
            score = 1.0 * (len(neighborsi & neighborsj)) / (len(neighborsi | neighborsj))
            scores.append(((node, n), score))
    scores=sorted(scores,key=lambda x: (-x[1],x[0][1]))
    return [scores[i] for i in range(0,k)]

def path_score(graph, root, k, beta, m):
    result ={}
    node2distances, node2num_paths, node2parents = bfs(graph,root,math.inf)
    for item in (set(graph.nodes())-set(graph.neighbors(root))):
      if item!=root:
        score = (beta**node2distances[item])*node2num_paths[item]
        result[(root,item)]=score
    result=sorted(result.items(),key = lambda x:(-x[1],x[0][0],x[0][1]))
    return result[:k]



def evaluate(predicted_edges, graph):
    e = []
    for x in predicted_edges:
        if graph.has_edge(x[0],x[1]) or graph.has_edge(x[1],x[0]):
            e.append(x)
    return (len(e)/len(predicted_edges))

def download_data():
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1, m=5)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
