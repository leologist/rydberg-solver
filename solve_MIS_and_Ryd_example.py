import numpy as np
import networkx as nx

from solve_MIS_and_Ryd import run_akmaxsat_for_Ryd_Ham,\
    run_MoMC_for_unweighted_MIS, graph_from_xy
    

def example_solve_MIS_and_Ryd(n=30, rho=2):
    """
    n = number of vertices
    rho = 2D density of vertices (# / area)
    """

    print("******************")
    print(f"Test solving MIS and Rydberg Hamiltonian for {n:d} atoms " +
          f"randomly distributed in a box with density {rho:0.3f}")

    # box dimenion
    L = np.sqrt(n/rho) 
    xy = np.random.random((n, 2))*L

    graph = graph_from_xy(xy, alpha=float('inf'))

    ## Uncomment below if want to draw graph
    # pos_dict = {i: xy[i] for i in range(xy.shape[0])}
    # nx.draw_networkx(graph, pos=pos_dict)

    MIS = run_MoMC_for_unweighted_MIS(graph)

    print(f'|MIS| = {len(MIS):d}')

    print("\n******************")
    print("******************\n")

    # Rydberg Hamiltonian is
    #    H = - \sum_i      n_i 
    #        + \sum_{<ij>} n_i n_j * V0 / (1 + ||r_i - r_j||^alpha)

    Ryd_GS = run_akmaxsat_for_Ryd_Ham(xy, V0=4, alpha=6)
    print(f'number of excited atoms in ground state = {len(Ryd_GS):d}')


if __name__ == "__main__":
    example_solve_MIS_and_Ryd()