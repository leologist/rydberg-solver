import os
import tempfile
import subprocess

import numpy as np
import networkx as nx

#####################
# Utility functions #
#####################

def run_cmd(cmd):
    proc = subprocess.Popen(cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr


def graph_from_xy(xy, alpha=6, threshold=1e-8):
    """ Create a networkx.Graph object from xy-coordinates
    where the edge weights correspond to
            1 / ||r_i - r_j||^alpha
    
    Args:
        xy = an n-by-2 matrix of xy-coordinates
        alpha = power parameter in interaction strength.
                Reduces to unit step function if alpha = float('inf')

    Returns:
        a networkx.Graph
    """
    n, d = xy.shape
    assert d == 2  # should be 2D coordinates
    
    def interaction(displacement):
        distnorm = np.linalg.norm(displacement)
        if alpha < float('inf'):
            return 1/distnorm**alpha
        elif distnorm <= 1:
            return 1
        else:
            return 0

    graph = nx.Graph()
    graph.add_nodes_from(list(range(n)))
    for i in range(n-1):
        for j in range(i+1, n):
            temp = interaction(xy[i]-xy[j])
            if temp >= threshold:
                graph.add_edge(i,j, weight=temp)

    return graph

#############################################################
# Solving unweighted Maximum Independent Set (MIS) problems #
#############################################################


def write_unweighted_MIS_to_ASCII(graph, file, node_to_index_map=None):
    """ Write MIS problem on a graph as a MaxClique problem
    in DIMACS ASCII form, to be passed to the MoMC algorithm.
    """
    gcomp = nx.complement(graph)
    file.write("p col %d %d\n" % (gcomp.number_of_nodes(), gcomp.number_of_edges()))

    if node_to_index_map is None:
        node_to_index_map = {v: i for i, v in enumerate(graph.nodes)}

    for u, v in gcomp.edges():
        # add 1 to node index to go from 0-based to 1-based index
        file.write("e %d %d\n" % (node_to_index_map[u]+1, node_to_index_map[v]+1))


def parse_MoMC_output(out):
    """ Read the output from MoMC in string and returns the list of
    nodes corresponding to the MIS (converted to zero-based index)
    """
    for line in out.splitlines():
        temp = str(line.decode())
        if temp.startswith('M'):
            return [int(x)-1 for x in temp[2:].split()]


def parse_MoMC_output_full(out):
    """ Read the output from MoMC in string and returns the list of
    nodes corresponding to the MIS (converted to zero-based index),
    and run-time and branching information.
    
    Returns:
        (MIS, Branching, Time, ProveBranching, ProveTime)
    """
    for line in out.splitlines():
        temp = str(line.decode())

        if temp.startswith('M'):
            MIS = [int(x)-1 for x in temp[2:].split()]

        if temp.startswith('s'):
            tempsplit = temp.split()
            assert len(MIS) == int(tempsplit[4])
            Branching = int(tempsplit[6])
            Time = float(tempsplit[8])
            ProveBranching = int(tempsplit[10])
            ProveTime = float(tempsplit[12])

    return MIS, Branching, Time, ProveBranching, ProveTime


###### Main Function for solving MIS ######

def run_MoMC_for_unweighted_MIS(graph: nx.Graph, node_to_index_map=None, verbose=True):
    """ Run the MoMC algorithm to find a MIS of a given graph.
    
    Requires the compiled executable MoMC to be in the
    current directory.
    
    Args:
        graph = a networkx.Graph whose MIS you want to find

    Returns:
        a list of nodes in the MIS
    """
    file_ID, filename = tempfile.mkstemp()
    if node_to_index_map is None:
        node_to_index_map = {v: i for i, v in enumerate(graph.nodes)}

    try:
        with os.fdopen(file_ID, 'w') as file:
            write_unweighted_MIS_to_ASCII(graph, file, node_to_index_map)

        code, out, err = run_cmd(['./MoMC', filename])

        if verbose:
            print("MoMC run status returncode = %d" % code)
            print("=========   MoMC output begins   =========")
            print(out.decode())
            print("=========   MoMC output ends     =========")

        if code < 0:
            print("=========   MoMC error message   =========")
            print(err.decode())

        MIS = parse_MoMC_output(out)
    finally:
        os.remove(filename)
        
    return MIS


def run_MoMC_for_unweighted_MIS_info(graph: nx.Graph, node_to_index_map=None, verbose=True):
    """ Run the MoMC algorithm to find a MIS of a given graph.
    
    Requires the compiled executable MoMC to be in the
    current directory.
    
    Args:
        graph = a networkx.Graph whose MIS you want to find

    Returns:
        (a list of nodes in the MIS, Branching, Time, ProveBranching, ProveTime)
    """
    file_ID, filename = tempfile.mkstemp()
    if node_to_index_map is None:
        node_to_index_map = {v: i for i, v in enumerate(graph.nodes)}

    try:
        with os.fdopen(file_ID, 'w') as file:
            write_unweighted_MIS_to_ASCII(graph, file, node_to_index_map)

        code, out, err = run_cmd(['./MoMC', filename])
        if verbose:
            print("MoMC run status returncode = %d" % code)
            print("=========   MoMC output begins   =========")
            print(out.decode())
            print("=========   MoMC output ends     =========")

        if code < 0:
            print("=========   MoMC error message   =========")
            print(err.decode())

        MIS, Branching, Time, ProveBranching, ProveTime = parse_MoMC_output_full(out)
    finally:
        os.remove(filename)
        
    return MIS, Branching, Time, ProveBranching, ProveTime


###################################################################
# Solving weighted Ising problems (including Rydberg Hamiltonian) #
###################################################################


def write_Ising_to_ASCII(h, J, file, precision=1e-6):
    r"""Write the Ising Hamiltonian minimization as a weighted
    max sat problem in conjunctive normal form, per DIMACS format
        (also see http://www.maxsat.udl.cat/14/requirements/)
    
    Ising problem: minimize the energy of 
    
        H = \sum_i h_i x_i + \sum_{i<j} J_{ij} x_i x_j
        
    where x_i are either 0 or 1

    This script convert H to the equivalent weighted CNF for MaxSAT,
    where each term is mapped to
        +x     => !x (true if x = 0)
        +x1 x2 => !x1 | !x2 (true iff x1 = x2 = 0)
        -x1 x2 => (x1 | x2) & (x1 | !x2) & (!x1 | x2) 
                  ^ 3 clauses are true if x1 = x2 = 1, 2 otherwise

    The weighted CNF is then written to the file, where h_i are J_{ij}
    are written up to the provided precision.
    """
    num_variables = len(h)
    
    h = np.round(np.array(h)/precision)
    
    J_ij = np.round(np.array(J[:, 2])/precision)
    num_clauses = np.count_nonzero(h)  \
                + np.count_nonzero(J_ij > 0) \
                + 3*np.count_nonzero(J_ij < 0)

    file.write("p wcnf %d %d\n" % (num_variables, num_clauses))
    
    for i in range(len(h)):
        if h[i] != 0:
            # add 1 to node index to go from 0-based to 1-based index
            file.write("%d %d 0\n" % (np.abs(h[i]), - np.sign(h[i])*(i+1)))

    for edge in range(J.shape[0]):
        # add 1 to node index to go from 0-based to 1-based index
        i = J[edge, 0] + 1
        j = J[edge, 1] + 1

        if J[edge, 2] > 0:
            file.write("%d %d %d 0\n" % (J_ij[edge], -i, -j))
        elif J[edge, 2] < 0:
            file.write("%d %d %d 0\n" % (-J_ij[edge], i, j))
            file.write("%d %d %d 0\n" % (-J_ij[edge], -i, j))
            file.write("%d %d %d 0\n" % (-J_ij[edge], i, -j))
        
        
def parse_akmaxsat_output(out):
    """ Read the output string from akmaxsat and returns a list of indices
    of nodes that are 1 instead of 0.
    """
    for line in out.splitlines():
        temp = str(line.decode())
        if temp.startswith('v'):
            GS = sorted([int(x) for x in temp[2:].split()], key=abs)
            return np.argwhere(np.array(GS)>0).flatten()


###### Main Function for solving Rydberg Hamiltonian ######

def run_akmaxsat_for_Ryd_Ham(xy, V0=4, alpha=6):
    r"""Find the ground state and its energy of the Rydberg Hamiltonian,
    using the akmaxsat algorithm (see Kugel)
        akmaxsat won many categories of MaxSat Evaluation 2010
    
        H = - \sum_i      n_i 
            + \sum_{<ij>} n_i n_j * V0 / ||r_i - r_j||^alpha
    
    Requires the compiled executable akmaxsat to be in the
    current directory.
    
    Args:
        xy = an n-by-2 matrix containing the xy-coordinates of atoms
        V0 = (optional) parameter setting the scale of interaction (default=4)
        alpha = (optional) the power law of interaction 1/r^alpha (default=6)
                uses unit-step function when alpha = float('inf')

    Returns:
        a list of 0-based indices of excited atoms (n_i=1) 
        in the lowest-energy state
    """

    graph = graph_from_xy(xy, alpha=alpha)
    h = - np.ones(graph.number_of_nodes())
    J = np.array([[u, v, V0*graph[u][v]['weight']] for u, v in graph.edges])
    
    file_ID, filename = tempfile.mkstemp()
    try:
        with os.fdopen(file_ID, 'w') as file:
            write_Ising_to_ASCII(h, J, file)

        code, out, err = run_cmd(['./akmaxsat', filename])
        print("akmaxsat run status returncode = %d" % code)

        print("========= akmaxsat output begins =========")
        print(out.decode())
        print("========= akmaxsat output ends   =========")

        if code < 0:
            print("========= akmaxsat error message =========")
            print(err.decode())

        GS = parse_akmaxsat_output(out)
    finally:
        os.remove(filename)
        
    n = graph.number_of_nodes()
    x_GS = np.zeros(n)
    x_GS[GS] = 1
    
    A = nx.adj_matrix(graph, nodelist=range(n))

    E_Ryd = -len(GS) + V0* np.vdot(x_GS, A*x_GS)/2
    
    print(f"Calculated Rydberg energy = {E_Ryd:0.6f}\n" +
          f" --------- max sat weight = {E_Ryd+n:0.6f}")
          
    return GS