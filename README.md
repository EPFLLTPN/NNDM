# NNDM
This repository contains an optimized, highly parallel Python implementation of the variational approach for open quantum systems using a neural network ansatz for sampling the non-equilibrium steady state and the dynamics of many-body open quantum systems.
For more information, see:

- A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
- A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020

## Functionality
The present version calculates the parameters (dynamical, steady-state) of an RBM ansatz for the density matrix with an extensive functionality:

#### Physical models: both in one- and two-dimensions
- dissipative XYZ model
- transverse field Ising model
- synthetic Ising model

#### Real-time dynamics
- first order Euler
- second order Runge-Kutta
#### Non-equilibrium steady state
- full sampling of <<L†L>>
- partial sampling of <<L†L>> (see Sec. 8.7.3)

The number of floating point operations to evaluate eq. (8.11) scales as N^3_p, if we assume that the number of Metropolis-Hastings steps N_MH is set to roughly the number of parameters N_p, as in Ref. [146]. The MCMC procedure also scales with the number of connected states Ncav , i.e. with the average number of non-zero elements in a column of the Lindbladian matrix. Therefore, the efficiency of the whole procedure scales as O(N^3_p + N_pN_cav ). In order to improve the computational efficiency, we have optimized the stochastic sampling, installed a parallel scheme by splitting the MCMC chain into
several independent threads and introduced GPU-accelerated linear algebra calculations.

## Some geography
Files are organized in the NNDM repository as follows:
 - `NNDM_spin/` root directory of the program. It contains the main file (Heisenberg_XYZ_Main.py) and the sub-folders.
 - `NNDM_spin/Classes/` contains the classes
 - `NNDM_spin/Tester/` contains files that can be used to benchmark with exact calculations when implementing a new physical model
- `Observable_Sampler_spin/` root directory of a program sampling expectation values of observables for a time-series of RBM parameters

## Requirements
### Python 3.6
NNDM is written in Python 3.6.
### Specific libraries
Numpy, scipy, cupy (for GPU calculations), lru (by amitdev).
### MPI
Since the parallelization is optimized for cluster computing, we used MVAPICH. The MVAPICH2 software, based on MPI 3.1 standard, delivers the best performance, scalability and fault tolerance for high-end computing systems and servers using InfiniBand, Omni-Path, Ethernet/iWARP, and RoCE networking technologies. 

# Usage
Example of a parallel run on a local machine:

`mpirun -n 4 python3.6 -u Heisenberg_XYZ_Main.py cpu lind_eu 2 2 1 1 1.1 1.2 1.0 1.0 0 10000 1500 0.2 0.000075 2 1000 '/path/to/input/input.txt' '/path/to/result/'`

where cpu/gpu is the flag for cpu or gpu calculations, lind_eu is the solver type which can be chosen as:
- lind_eu : Lindbladian Euler
- lind_runge : Lindbladian Runge Kutta
- ldag_full : full sampling L+L
- ldag_y : just a predefined number drawn
then model parameters as N_x, N_y (spin number is x/y direction), hidden node ratio, mixing node ratio, J_x, J_y, J_z, decay rate, external field, iterations, Monte Carlo sweep, thermal rate, learning rate, number drawn in case of method ldag_y, number of stored density matrix elements in dictionary (to keep in memory the frequently visited elements), input file (in case of restarting a simulation), result folder. The results of the simulations then will be printed in '/path/to/result/'. It is necessary that the folder exists.

An example input file can be found in /NNDM_spin/example_input.txt

Example of a parallel run file on a computational cluster using a SLURM workload manager:

`#!/bin/bash`

`#SBATCH --nodes 1`

`#SBATCH --ntasks 8`

`#SBATCH --cpus-per-task 1`

`#SBATCH --time 10:00:00`

`#SBATCH --account=ltpn`

`module purge`

`module load gcc python mvapich2`

`srun python /path/to/main/Heisenberg_XYZ_Main.py 2 2 2 2 0.9 1.1 1 1 0 15000 2000 0.2 0.001 1000 /path/to/input/input.txt /path/to/output/`

# License
Except where otherwise stated, HANDE is under the 'Lesser GNU Public License v3.0'.
