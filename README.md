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
