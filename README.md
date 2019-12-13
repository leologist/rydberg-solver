# rydberg-solver
solvers for finding MIS and ground state of Rydberg Hamiltonian

## Solvers
1. `MoMC` - a branch-and-bound algorithm for solving MIS. See [https://home.mis.u-picardie.fr/~cli/EnglishPage.html](https://home.mis.u-picardie.fr/~cli/EnglishPage.html).
2. `akmaxsat` - a complete solver for MaxSAT, guaranteed to find the optimal assignment. Winner of [2010 MaxSAT Evaluation competition](http://www.maxsat.udl.cat/10/results/#wms-random). This can be applied to any Ising problem for energy minimization, including the Rydberg Hamiltonian.

## Compiling
Before using, must first compile the solvers for use. Simply run

`./build_solvers.sh`

## Usage

If your problem (MIS or Rydberg Hamiltonian) is most easily accessible in python, it's recommended to use the python utilties in `solve_MIS_and_Ryd.py` to apply the solvers.
See `solve_MIS_and_Ryd_example.py` for example

Alternatively, one can write the problem information in the appropriate DIMACS format, and then use the solvers directly by

`./MoMC path_to_instance_file.txt`
`./akmaxsat path_to_instance_file.txt`
