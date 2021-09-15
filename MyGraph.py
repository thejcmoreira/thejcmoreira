####################################################################################################################
###                                    Algorithms for Bioinformatics
###                                 ***  Week 10 - Exercise 8 – Graphs and Biological Networks ***
###
### Date: 10 to 26 May 2021
###
### Group D
### Student: Joel Cláudio Pinto Moreira  Number: 202008682
### Student: Luiz Claudio Navarro        Number: 202009075
### Student: Tiago Filipe dos Santos     Number: 202008971
###
####################################################################################################################
## Complete the code and develop the functions in the MyGraph.py. Submit a unique python file in
## the Moodle. When running as: python MyGraph.py output the request functionalities.
####################################################################################################################
## Graph represented as adjacency list using a dictionary
## keys are vertices
## values of the dictionary represent the list of adjacent vertices of the key node

class MyGraph:
    def __init__(self, g={}):
        ''' Constructor - takes dictionary to fill the graph as input; default is empty dictionary '''
        self.graph = g

    def print_graph(self):
        ''' Prints the content of the graph as adjacency list '''
        for v in self.graph.keys():
            print(v, " -> ", self.graph[v])

    ## get basic info
    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . get_nodes(self): Returns list of nodes in the graph
    # ****************************************************************************************
    def get_nodes(self):
        ''' Returns list of nodes in the graph '''
        return list(self.graph.keys())

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . get_edges(self): Returns edges in the graph as a list of tuples (origin, destination)
    # ****************************************************************************************
    def get_edges(self):
        ''' Returns edges in the graph as a list of tuples (origin, destination) '''
        edges = []
        for origin_node, glist in self.graph.items():
            for dest_node in glist:
                edges.append((origin_node, dest_node))
        return edges

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . size(self): Returns size of the graph : number of nodes, number of edges
    # ****************************************************************************************
    def size(self):
        ''' Returns size of the graph : number of nodes, number of edges '''
        return len(self.get_nodes()), len(self.get_edges())

    ## add nodes and edges
    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . add_vertex(self, v): Add a vertex to the graph; tests if vertex exists not adding if it does
    # ***************************************************************************************
    def add_vertex(self, v):
        ''' Add a vertex to the graph; tests if vertex exists not adding if it does '''
        if v not in self.graph.keys():
            self.graph[v] = []
        else:
            print(v, "already exists on this graph.")

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . add_edge(self, o, d): Add edge to the graph; if ver6ces do not exist, they are
    # added to the graph
    # ****************************************************************************************
    def add_edge(self, o, d):
        ''' Add edge to the graph; if vertices do not exist, they are added to the graph '''
        exists = False
        for etuple in self.get_edges():
            if etuple == (o, d):
                exists = True
        if not exists:
            self.graph[o].append(d)
        else:
            print(o, "is already connected to", d, "on this graph.")

    ## successors, predecessors, adjacent nodes

    def get_successors(self, v):
        return list(self.graph[v])  # needed to avoid list being overwritten of result of the function is used

    def get_predecessors(self, v):
        res = []
        for k in self.graph.keys():
            if v in self.graph[k]:
                res.append(k)
        return res

    def get_adjacents(self, v):
        suc = self.get_successors(v)
        pred = self.get_predecessors(v)
        res = pred
        for p in suc:
            if p not in res: res.append(p)
        return res

    ## degrees

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . out_degree(self, v): Number of successors of vertex
    # ****************************************************************************************
    def out_degree(self, v):
        """Calculates the number of successors of vertex """
        return len(self.get_successors(v))

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . in_degree(self, v): Number of predecessors of vertex
    # ****************************************************************************************
    def in_degree(self, v):
        """ Calculates the number of predecessors of vertex """
        return len(self.get_predecessors(v))

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . degree(self, v): Unique set of predecessors and successors of vertex
    # ****************************************************************************************
    def degree(self, v):
        """ Unique set of predecessors and successors of vertex """
        return len(self.get_adjacents(v))

    def all_degrees(self, deg_type="inout"):
        ''' Computes the degree (of a given type) for all nodes.
        deg_type can be "in", "out", or "inout" '''
        degs = {}
        for v in self.graph.keys():
            if deg_type == "out" or deg_type == "inout":
                degs[v] = len(self.graph[v])
            else:
                degs[v] = 0
        if deg_type == "in" or deg_type == "inout":
            for v in self.graph.keys():
                for d in self.graph[v]:
                    if deg_type == "in" or v not in self.graph[d]:
                        degs[d] = degs[d] + 1
        return degs

    def highest_degrees(self, all_deg=None, deg_type="inout", top=10):
        if all_deg is None:
            all_deg = self.all_degrees(deg_type)
        ord_deg = sorted(list(all_deg.items()), key=lambda x: x[1], reverse=True)
        return list(map(lambda x: x[0], ord_deg[:top]))

    ## topological metrics over degrees
    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . mean_degree(self, deg_type = "inout"): average degree of all nodes: sum of all degrees
    # divided by number of nodes
    # ****************************************************************************************
    def mean_degree(self, deg_type="inout"):
        ''' Returns the averages degree of all nodes: sum of all degrees divided by number of nodes'''
        total = 0
        for k, v in self.all_degrees(deg_type=deg_type).items():
            total += v
        return total / self.size()[0]

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . prob_degree(self, deg_type = "inout"): Counting of the number of occurrences of each
    # degree in the network and its frequencies;
    # ****************************************************************************************
    def prob_degree(self, deg_type="inout"):
        """ Counts the number of occurrences of each degree in the network and its frequencies """
        prob_dict = {}
        for k, v in self.all_degrees(deg_type=deg_type).items():
            if v not in prob_dict:
                prob_dict[v] = 1
            else:
                prob_dict[v] += 1
        return prob_dict

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . print_prob_degree(self, counts): Print the degrees and frequencies one per line;
    # ****************************************************************************************
    def print_prob_degree(self, prob_dict=None):
        """ Prints the degrees and frequencies one per line """
        if prob_dict is None:
            prob_dict = self.prob_degree()
        for k, v in prob_dict.items():
            print("Degree", k, "\t", v)

    ## BFS and DFS searches

    def reachable_bfs(self, v):
        l = [v]  # list of nodes to be handled
        res = []  # list of nodes to return the result
        while len(l) > 0:
            node = l.pop(0)  # implements a queue: LILO
            if node != v: res.append(node)
            for elem in self.graph[node]:
                if elem not in res and elem not in l and elem != node:
                    l.append(elem)
        return res

    def reachable_dfs(self, v):
        l = [v]
        res = []
        while len(l) > 0:
            node = l.pop(0)  # implements a stack:
            if node != v: res.append(node)
            s = 0
            for elem in self.graph[node]:
                if elem not in res and elem not in l:
                    l.insert(s, elem)
                    s += 1
        return res

    def distance(self, s, d):
        if s == d: return 0
        l = [(s, 0)]
        visited = [s]
        while len(l) > 0:
            node, dist = l.pop(0)
            for elem in self.graph[node]:
                if elem == d:
                    return dist + 1
                elif elem not in visited:
                    l.append((elem, dist + 1))
                    visited.append(elem)
        return None

    def shortest_path(self, s, d):
        if s == d: return 0
        l = [(s, [])]
        visited = [s]
        while len(l) > 0:
            node, preds = l.pop(0)
            for elem in self.graph[node]:
                if elem == d:
                    return preds + [node, elem]
                elif elem not in visited:
                    l.append((elem, preds + [node]))
                    visited.append(elem)
        return None

    ## clustering

    def clustering_coef(self, v):
        adjs = self.get_adjacents(v)
        if len(adjs) <= 1: return 0.0
        # calculate the number of links of the adjacent nodes
        ligs = 0
        # compare pairwisely if nodes in this list are connected between them
        for i in adjs:
            for j in adjs:
                if i != j:
                    # check if i and j are connected to each other; if yes increment counter of links
                    if j in self.get_adjacents(i):
                        ligs += 1
        return float(ligs) / (len(adjs) * (len(adjs) - 1))

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . all_clustering_coefs(self): Returns the clustering coefficient for all the nodes in the
    # network;
    # ****************************************************************************************
    def all_clustering_coefs(self):
        """ Returns the clustering coefficient for all the nodes in the network """
        # go through all the nodes and calculate its cc
        # put those in a dictionary and return
        all_cc = {}
        for v in self.get_nodes():
            all_cc[v] = self.clustering_coef(v)
        return all_cc

    # ***************************************************************************************
    # Task 1 Complete the remaining methods
    # . mean_clustering_coef(self): Calculates the mean clustering coefficient for the network
    # ****************************************************************************************
    def mean_clustering_coef(self):
        """ Calculates the mean clustering coefficient for the network """
        total = 0
        for k, v in self.all_clustering_coefs().items():
            total += v
        return total / self.size()[0]

    # ***************************************************************************************
    # Task 2 Gene Coexpression Network
    # The file m_lung_gexp.tab contains a table that represents the correlation between 200
    # genes. These correspond to genes where the expression levels were measured in the human
    # lung tissue across 487 samples. The first and second column of the table represent the gene
    # names and the third column represent the Pearson correlation between these two genes
    # across the 487 samples.
    # ***************************************************************************************
    # ***************************************************************************************
    # 2.1 Create a method to add to class MyGraph called create_network_from_file that reads
    # the data in m_lung_gexp.tab and creates a network. Nodes correspond to genes, i.e.
    # elements from column Gene1 and Gene2. An edge corresponds to a pair of genes (each
    # row). The method receives a second parameter called min_correlation. Only pair of genes
    # with an absolute minimal correlation should be considered as edges, i.e. :
    # for the tuple (Gi, Gj, corr), (Gi,Gj) is an edge if: |corr| > min_correlation
    # ***************************************************************************************
    def create_network_from_file(self, fname, min_correlation):
        ctlin = 0
        print("*** Reading genes correlation file {:s} with min_corr {:f} ***".
              format(fname, min_correlation))
        with open(fname, "r") as tsvfile:
            row = tsvfile.readline()
            while len(row) > 0:
                t1 = row.find("\t")
                if t1 <= 0:  # if 0 starts with tab, if < 0 there is no tab
                    print("Discarded line invalid first field:", row)
                else:
                    t2 = row.find("\t", t1+1)
                    if t2 <= 0:  # if 0 then there is tab tab, if < 0 there is no second tab
                        print("Discarded line invalid second field:", row)
                    else:
                        if ctlin == 0:
                            pass
                        else:
                            g1 = row[:t1]
                            g2 = row[t1+1:t2]
                            try:
                                cc = float(row[t2+1:])
                            except ValueError:
                                print("Discarded line invalid third field:", row)
                            else:
                                if abs(cc) > min_correlation:
                                    if g1 not in self.graph.keys():
                                        self.add_vertex(g1)
                                    if g2 not in self.graph.keys():
                                        self.add_vertex(g2)
                                    if g1 != g2:  # if correlation between same node do not put edge
                                        self.add_edge(g1, g2)
                ctlin += 1
                row = tsvfile.readline()
        print("{:d} lines read from file {:s}.".format(ctlin, fname))
        return

    # Characterisation of the Gene Correlation Network
    # 2.2 Given the available methods obtain the statistics to create a characterisation of the
    # network features. Output as tabular format (not necessary to use external modules as the
    # script should run without any additional module installation) the statistics values, with a
    # column for the feature/statistic and another column for the value found for the network.
    # ***************************************************************************************
    def coef_degree(self, deg_type="inout"):
        """ Compute the average clustering coefficient for nodes of same degree in the network """
        degr_dict = {}
        coef_dict = {}
        for vert, deg in self.all_degrees(deg_type=deg_type).items():
            if deg not in degr_dict:
                degr_dict[deg] = [vert]
            else:
                degr_dict[deg].append(vert)
        for deg in sorted(degr_dict.keys()):
            tot_coef = 0
            vert_list = degr_dict[deg]
            for vert in vert_list:
                tot_coef += self.clustering_coef(vert)
            coef_dict[deg] = tot_coef / len(vert_list)
        return coef_dict


    def print_statistics(self, direct=False):
        tabfmt = "{:36s} {:13s}"
        print(tabfmt.format("------- feature/statistic --------", "--- value ---"))
        n_v, n_e = self.size()
        print(tabfmt.format("Number of vertices", str(n_v)))
        if direct:
            print(tabfmt.format("Number of edges (direct)", str(n_e)))
        else:
            print(tabfmt.format("Number of edges (non-direct)", str(int(n_e / 2))))
        print(tabfmt.format("Average degree", str(self.mean_degree())))
        print(tabfmt.format("Average clustering coefficient", str(self.mean_clustering_coef())))
        prob_dict = self.prob_degree()
        coef_dict = self.coef_degree()
        tabfmt = "{:12s} {:8d} {:8.2f}% {:8.4f}"
        print("---Degree--- --Freq-- --Prob%-- --Coef--")
        for deg in sorted(prob_dict.keys()):
            freq = prob_dict[deg]
            coef = coef_dict[deg]
            print(tabfmt.format("  degree {:d}".format(deg), freq, freq / n_v * 100.0, coef))
        return

    # ***************************************************************************************
    # 2.3 Adapt the function reachable_bfs to a function called reachable_bfs_with_distance that
    # for each node also prints the corresponding distance. The output should be a list of tuples
    # with nodes and distance as: [(node, distance)]. Use as example the graph from the class
    # exercise and run for the TP53 gene.
    # ***************************************************************************************
    def reachable_bfs_with_distance(self, v):
        l = [(v,0)]  # list of nodes to be handled
        res = []  # list of nodes to return the result
        while len(l) > 0:
            node, dist = l.pop(0)  # implements a queue: LILO
            if node != v: res.append((node, dist))
            dist += 1
            for elem in self.graph[node]:
                if elem != node:
                    exists = False
                    for nd, dst in res:
                        if elem == nd:
                            exists = True
                            break
                    if not exists:
                        for nd, dst in l:
                            if elem == nd:
                                exists = True
                                break
                        if not exists:
                            l.append((elem, dist))
        return res

def test_aula():
    print("--- Class work exercise ---")
    gr_ta = MyGraph({})
    gr_ta.add_vertex("BRAF")
    gr_ta.add_vertex("NF1")
    gr_ta.add_vertex("NRAS")
    gr_ta.add_vertex("ERBB3")
    gr_ta.add_vertex("FBXW7")
    gr_ta.add_vertex("FLT3")
    gr_ta.add_vertex("PTEN")
    gr_ta.add_vertex("PIK3CA")
    gr_ta.add_vertex("DNMT3A")
    gr_ta.add_vertex("TP53")
    gr_ta.add_vertex("CTNNB1")
    gr_ta.add_vertex("APC")
    gr_ta.add_vertex("LPHN2")
    gr_ta.add_vertex("SF3B1")
    gr_ta.add_vertex("SMAD4")
    gr_ta.add_vertex("NCOR1")

    gr_ta.add_edge("BRAF","NRAS")
    gr_ta.add_edge("NRAS", "BRAF")

    gr_ta.add_edge("NF1","NRAS")
    gr_ta.add_edge("NRAS", "NF1")

    gr_ta.add_edge("NRAS","ERBB3")
    gr_ta.add_edge("ERBB3", "NRAS")

    gr_ta.add_edge("NRAS","PIK3CA")
    gr_ta.add_edge("PIK3CA", "NRAS")

    gr_ta.add_edge("NRAS", "FLT3")
    gr_ta.add_edge("FLT3", "NRAS")

    gr_ta.add_edge("ERBB3","PIK3CA")
    gr_ta.add_edge("PIK3CA", "ERBB3")

    gr_ta.add_edge("FLT3","PIK3CA")
    gr_ta.add_edge("PIK3CA", "FLT3")

    gr_ta.add_edge("PTEN","PIK3CA")
    gr_ta.add_edge("PIK3CA", "PTEN")

    gr_ta.add_edge("PTEN","TP53")
    gr_ta.add_edge("TP53", "PTEN")

    gr_ta.add_edge("PIK3CA","TP53")
    gr_ta.add_edge("TP53", "PIK3CA")

    gr_ta.add_edge("PIK3CA","CTNNB1")
    gr_ta.add_edge("CTNNB1", "PIK3CA")

    gr_ta.add_edge("CTNNB1","APC")
    gr_ta.add_edge("APC", "CTNNB1")

    gr_ta.add_edge("CTNNB1","SMAD4")
    gr_ta.add_edge("SMAD4", "CTNNB1")

    gr_ta.add_edge("SMAD4","NCOR1")
    gr_ta.add_edge("NCOR1", "SMAD4")

    gr_ta.print_graph()
    nodes, edges = gr_ta.size()
    # As non-direct graph, then edges are duplicated in both directions
    print("Nodes = {:d}, Edges = {:d}".format(nodes, int(edges / 2)))

    print("Adjacents to NRAS = ", gr_ta.get_adjacents("NRAS"))

    highnode = gr_ta.highest_degrees()[0]
    adjhigh = gr_ta.get_adjacents(highnode)
    print("Highest degree node =", highnode, ", degree =",
          len(adjhigh), ", adjacents =", adjhigh)

    print("Degree distribution")
    gr_ta.print_prob_degree()

    print("Length of shortest path between TP53 and NF1:", gr_ta.distance("TP53", "NF1"))
    for gene in ["PIK3CA", "NF1", "SMAD4"]:
        print("DFS traversal from", gene, ":", gr_ta.reachable_dfs(gene))
        print("BFS traversal from", gene, ":", gr_ta.reachable_bfs(gene))

    return gr_ta


if __name__ == "__main__":
    print("------------------------------- Begin Exercise 8 - Group D --------------------------------")
    print()
    print("*******************************************************************************************")
    print("*** Task 1 - Test Graph 1                                                               ***")
    print("*******************************************************************************************")
    gr = MyGraph({})
    gr.add_vertex(1)
    gr.add_vertex(2)
    gr.add_vertex(3)
    gr.add_vertex(4)
    gr.add_edge(1, 2)
    gr.add_edge(2, 3)
    gr.add_edge(3, 2)
    gr.add_edge(3, 4)
    gr.add_edge(4, 2)
    gr.print_graph()
    print("Size (vertices, edges)", gr.size())
    print()
    print("Sucessors of vertex 2:", gr.get_successors(2))
    print("Predecessors of vertex 2:", gr.get_predecessors(2))
    print("Adjacents of vertex 2:", gr.get_adjacents(2))
    print()
    print("In degree of vertex 2:", gr.in_degree(2))
    print("Out degree of vertex 2:", gr.out_degree(2))
    print("Degree of vertex 2:", gr.degree(2))
    print()
    print("All degrees:", gr.all_degrees("inout"))
    print("All in degrees:", gr.all_degrees("in"))
    print("All out degrees:", gr.all_degrees("out"))
    print()

    print("Mean degrees:", gr.mean_degree())
    print("Prob degrees:", gr.prob_degree())
    gr.print_prob_degree()

    print()
    # print(gr.mean_distances())
    print("Clustering coef of vertex 1:", gr.clustering_coef(1))
    print("Clustering coef of vertex 2:", gr.clustering_coef(2))

    print("All Clustering coefs:", gr.all_clustering_coefs())
    print("Mean of clustering coefs:", gr.mean_clustering_coef())

    print()
    print("*******************************************************************************************")
    print("*** Task 1 - Test Graph 2                                                               ***")
    print("*******************************************************************************************")
    gr2 = MyGraph({1:[2,3,4], 2:[5,6],3:[6,8],4:[8],5:[7],6:[],7:[],8:[]})
    gr2.print_graph()
    print("Size (vertices, edges)", gr2.size())
    print("Reacheable vertices from vertex 1 using BFS:", gr2.reachable_bfs(1))
    print("Reacheable vertices from vertex 1 using DFS:", gr2.reachable_dfs(1))
    print()
    print("Distance from 1 to 7:", gr2.distance(1,7))
    print("Shortest path from 1 to 7:", gr2.shortest_path(1,7))
    print()
    print("Distance from 1 to 8:", gr2.distance(1,8))
    print("Shortest path from 1 to 8:", gr2.shortest_path(1,8))
    print()
    print("Distance from 6 to 1:", gr2.distance(6,1))
    print("Shortest path from 6 to 1:", gr2.shortest_path(6,1))
    print()
    print("Reacheable vertices from vertex 1 using BFS with distance:",
          gr2.reachable_bfs_with_distance(1))

    print()
    print("*******************************************************************************************")
    print("*** Task 2 - Exercises 2.1, 2.2 - Tests using m_lung_gexp.tab file                      ***")
    print("*******************************************************************************************")
    gr3 = MyGraph({})
    gr3.print_graph()
    print("Loading file m_lung_gexp.tab which has already edges in both directions (Non-Direct)")
    gr3.create_network_from_file("m_lung_gexp.tab", 0.5)
    gr3.print_statistics(direct=False)

    print(" ")
    print("*******************************************************************************************")
    print("*** Task 2 - Exercises 2.3 - Tests of reachable_bfs_with_distance with class work graph ***")
    print("*******************************************************************************************")
    gr_aula = test_aula()
    print()
    print("Reacheable vertices from gene TP53 using DFS:",
          gr_aula.reachable_dfs("TP53"))
    print("Reacheable vertices from gene TP53 using BFS:",
          gr_aula.reachable_bfs("TP53"))
    print()
    print("Reacheable vertices from gene TP53 using BFS with distance:",
          gr_aula.reachable_bfs_with_distance("TP53"))

    print()
    print("-------------------------------- End Exercise 8 - Group D ---------------------------------")

    print()
    print("*******************************************************************************************")
    print("*** Task 2 - Exercises 2.4 - Analysis of graph from m_lung_gexp.tab file ***")
    print("*******************************************************************************************")
    # ***************************************************************************************
    # 2.4 For the the three types of networks, discuss which type of network this one is more
    # similar. Justify your answer. Write a text as comment in the script.
    # ***************************************************************************************
    print("2.4 ANSWER according statistics above of m_lung_gexp network :")
    print("The network for genes resulting of m_lung_gexp.tab file, filtered with")
    print("Pearson correlation coefficients > 0.5, is type: SCALE-FREE")
    print("Because P(k) is exponencially decreasing and C(k) is almost constant.")

#######################################################################################################
###                                     Test Execution Results                                      ###
#######################################################################################################
# ------------------------------- Begin Exercise 8 - Group D --------------------------------
#
# *******************************************************************************************
# *** Task 1 - Test Graph 1                                                               ***
# *******************************************************************************************
# 1  ->  [2]
# 2  ->  [3]
# 3  ->  [2, 4]
# 4  ->  [2]
# Size (vertices, edges) (4, 5)
#
# Sucessors of vertex 2: [3]
# Predecessors of vertex 2: [1, 3, 4]
# Adjacents of vertex 2: [1, 3, 4]
#
# In degree of vertex 2: 3
# Out degree of vertex 2: 1
# Degree of vertex 2: 3
#
# All degrees: {1: 1, 2: 3, 3: 2, 4: 2}
# All in degrees: {1: 0, 2: 3, 3: 1, 4: 1}
# All out degrees: {1: 1, 2: 1, 3: 2, 4: 1}
#
# Mean degrees: 2.0
# Prob degrees: {1: 1, 3: 1, 2: 2}
# Degree 1         1
# Degree 3         1
# Degree 2         2
#
# Clustering coef of vertex 1: 0.0
# Clustering coef of vertex 2: 0.3333333333333333
# All Clustering coefs: {1: 0.0, 2: 0.3333333333333333, 3: 1.0, 4: 1.0}
# Mean of clustering coefs: 0.5833333333333333
#
# *******************************************************************************************
# *** Task 1 - Test Graph 2                                                               ***
# *******************************************************************************************
# 1  ->  [2, 3, 4]
# 2  ->  [5, 6]
# 3  ->  [6, 8]
# 4  ->  [8]
# 5  ->  [7]
# 6  ->  []
# 7  ->  []
# 8  ->  []
# Size (vertices, edges) (8, 9)
# Reacheable vertices from vertex 1 using BFS: [2, 3, 4, 5, 6, 8, 7]
# Reacheable vertices from vertex 1 using DFS: [2, 5, 7, 6, 3, 8, 4]
#
# Distance from 1 to 7: 3
# Shortest path from 1 to 7: [1, 2, 5, 7]
#
# Distance from 1 to 8: 2
# Shortest path from 1 to 8: [1, 3, 8]
#
# Distance from 6 to 1: None
# Shortest path from 6 to 1: None
#
# Reacheable vertices from vertex 1 using BFS with distance: [(2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (8, 2), (7, 3)]
#
# *******************************************************************************************
# *** Task 2 - Exercises 2.1, 2.2 - Tests using m_lung_gexp.tab file                      ***
# *******************************************************************************************
# Loading file m_lung_gexp.tab which has already edges in both directions (Non-Direct)
# *** Reading genes correlation file m_lung_gexp.tab with min_corr 0.500000 ***
# 40001 lines read from file m_lung_gexp.tab.
# ------- feature/statistic --------   --- value ---
# Number of vertices                   200
# Number of edges (non-direct)         1936
# Average degree                       19.36
# Average clustering coefficient       0.43522536725355904
# ---Degree--- --Freq-- --Prob%-- --Coef--
#   degree 0         35    17.50%   0.0000
#   degree 1         22    11.00%   0.0000
#   degree 2         14     7.00%   0.7143
#   degree 3          4     2.00%   0.4167
#   degree 4          7     3.50%   0.5952
#   degree 5          3     1.50%   0.5333
#   degree 6          1     0.50%   0.5333
#   degree 7          5     2.50%   0.5905
#   degree 8          2     1.00%   0.7321
#   degree 9          5     2.50%   0.6278
#   degree 10         4     2.00%   0.6500
#   degree 11         1     0.50%   0.2909
#   degree 12         1     0.50%   0.9697
#   degree 13         3     1.50%   0.5256
#   degree 14         3     1.50%   0.5201
#   degree 16         2     1.00%   0.7500
#   degree 17         4     2.00%   0.7426
#   degree 19         4     2.00%   0.6213
#   degree 20         3     1.50%   0.3158
#   degree 21         1     0.50%   0.7857
#   degree 22         2     1.00%   0.5931
#   degree 23         2     1.00%   0.5553
#   degree 24         1     0.50%   0.6703
#   degree 26         3     1.50%   0.5426
#   degree 27         4     2.00%   0.6538
#   degree 28         3     1.50%   0.6825
#   degree 29         1     0.50%   0.5493
#   degree 30         1     0.50%   0.8575
#   degree 31         2     1.00%   0.6581
#   degree 32         2     1.00%   0.5454
#   degree 33         1     0.50%   0.5871
#   degree 34         1     0.50%   0.4884
#   degree 35         3     1.50%   0.6543
#   degree 36         3     1.50%   0.5921
#   degree 38         1     0.50%   0.6785
#   degree 39         3     1.50%   0.6761
#   degree 40         1     0.50%   0.6859
#   degree 41         2     1.00%   0.6372
#   degree 43         2     1.00%   0.6440
#   degree 44         3     1.50%   0.5585
#   degree 45         3     1.50%   0.5754
#   degree 46         1     0.50%   0.6705
#   degree 47         3     1.50%   0.5865
#   degree 48         3     1.50%   0.5733
#   degree 49         3     1.50%   0.6131
#   degree 52         3     1.50%   0.6302
#   degree 53         1     0.50%   0.5726
#   degree 55         1     0.50%   0.6377
#   degree 56         1     0.50%   0.6162
#   degree 57         1     0.50%   0.5714
#   degree 58         1     0.50%   0.6007
#   degree 59         2     1.00%   0.5994
#   degree 60         1     0.50%   0.5729
#   degree 62         5     2.50%   0.5523
#   degree 63         2     1.00%   0.5604
#   degree 64         2     1.00%   0.5216
#   degree 65         2     1.00%   0.5087
#
# *******************************************************************************************
# *** Task 2 - Exercises 2.3 - Tests of reachable_bfs_with_distance with class work graph ***
# *******************************************************************************************
# --- Class work exercise ---
# BRAF  ->  ['NRAS']
# NF1  ->  ['NRAS']
# NRAS  ->  ['BRAF', 'NF1', 'ERBB3', 'PIK3CA', 'FLT3']
# ERBB3  ->  ['NRAS', 'PIK3CA']
# FBXW7  ->  []
# FLT3  ->  ['NRAS', 'PIK3CA']
# PTEN  ->  ['PIK3CA', 'TP53']
# PIK3CA  ->  ['NRAS', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'CTNNB1']
# DNMT3A  ->  []
# TP53  ->  ['PTEN', 'PIK3CA']
# CTNNB1  ->  ['PIK3CA', 'APC', 'SMAD4']
# APC  ->  ['CTNNB1']
# LPHN2  ->  []
# SF3B1  ->  []
# SMAD4  ->  ['CTNNB1', 'NCOR1']
# NCOR1  ->  ['SMAD4']
# Nodes = 16, Edges = 14
# Adjacents to NRAS =  ['BRAF', 'NF1', 'ERBB3', 'FLT3', 'PIK3CA']
# Highest degree node = PIK3CA , degree = 6 , adjacents = ['NRAS', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'CTNNB1']
# Degree distribution
# Degree 1         4
# Degree 5         1
# Degree 2         5
# Degree 0         4
# Degree 6         1
# Degree 3         1
# Length of shortest path between TP53 and NF1: 3
# DFS traversal from PIK3CA : ['NRAS', 'BRAF', 'NF1', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'CTNNB1', 'APC', 'SMAD4', 'NCOR1']
# BFS traversal from PIK3CA : ['NRAS', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'CTNNB1', 'BRAF', 'NF1', 'APC', 'SMAD4', 'NCOR1']
# DFS traversal from NF1 : ['NRAS', 'BRAF', 'ERBB3', 'PIK3CA', 'PTEN', 'TP53', 'CTNNB1', 'APC', 'SMAD4', 'NCOR1', 'FLT3']
# BFS traversal from NF1 : ['NRAS', 'BRAF', 'ERBB3', 'PIK3CA', 'FLT3', 'PTEN', 'TP53', 'CTNNB1', 'APC', 'SMAD4', 'NCOR1']
# DFS traversal from SMAD4 : ['CTNNB1', 'PIK3CA', 'NRAS', 'BRAF', 'NF1', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'APC', 'NCOR1']
# BFS traversal from SMAD4 : ['CTNNB1', 'NCOR1', 'PIK3CA', 'APC', 'NRAS', 'ERBB3', 'FLT3', 'PTEN', 'TP53', 'BRAF', 'NF1']
#
# Reacheable vertices from gene TP53 using DFS: ['PTEN', 'PIK3CA', 'NRAS', 'BRAF', 'NF1', 'ERBB3', 'FLT3', 'CTNNB1', 'APC', 'SMAD4', 'NCOR1']
# Reacheable vertices from gene TP53 using BFS: ['PTEN', 'PIK3CA', 'NRAS', 'ERBB3', 'FLT3', 'CTNNB1', 'BRAF', 'NF1', 'APC', 'SMAD4', 'NCOR1']
#
# Reacheable vertices from gene TP53 using BFS with distance: [('PTEN', 1), ('PIK3CA', 1), ('NRAS', 2), ('ERBB3', 2), ('FLT3', 2), ('CTNNB1', 2), ('BRAF', 3), ('NF1', 3), ('APC', 3), ('SMAD4', 3), ('NCOR1', 4)]
#
# *******************************************************************************************
# *** Task 2 - Exercises 2.4 - Analysis of graph from m_lung_gexp.tab file ***
# *******************************************************************************************
# 2.4 ANSWER according statistics above of m_lung_gexp network :
# The network for genes resulting of m_lung_gexp.tab file, filtered with
# Pearson correlation coefficients > 0.5, is type: SCALE-FREE
# Because P(k) is exponencially decreasing and C(k) is almost constant.
#
# -------------------------------- End Exercise 8 - Group D ---------------------------------
#
