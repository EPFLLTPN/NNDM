 # 
 # This file is part of the EPFLLTPN/NNDM distribution (https://github.com/EPFLLTPN/NNDM).
 # Copyright (c) 2019 Alexandra Nagy.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #
 # Heisenberg_XYZ_Main.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

# Main file to run the NQS for the Heisenberg 1D XYZ model

from Classes import NQS, NQS_nohidden
from Classes import SR, SR_Runge
from Classes import Heisenberg_2D_XYZ, Transverse_Ising_1D, Heisenberg_Carleo, Transverse_Ising_2D, Heisenberg_2D_XYZ_Hz, Rydberg_1D, Rydberg_2D_open
from Classes import MH_ldag_y, MH_ldag_full, MH_lind
from Classes import IO, IO_nohidden
 
from Tester import MatlabTester
from mpi4py import MPI
from lru import LRU
import sys
import time
import re

# IMPORTANT NOTICE IN IO

# LRU implementation is done by 'amitdev' - github

##################################################################################################
# Type of method:
# 
#  - lind_eu : Lindbladian Euler
#  - lind_runge : Lindbladian Runge Kutta
#  - ldag_full : full sampling L+L
#  - ldag_y : just a predefined number drawn
# 
# All parameters need an input value. Depending on the model, some will be ignored by the code
##################################################################################################

# method, N_x, N_y, hidden_self_ratio, mixing_ratio Jx, Jy, Jz, gamma, field, iteration, MC_sweep, thermal_rate, learning_rate, lru size, drawn_sample, restart, path

# choice between gpu and cpu. it also automatically converts to cpu if cuda is missing

# if NOT restart, use 'none'. Like this. Not None.
 

def main():
    
    then = time.time();
    #######################################################################
    # Parameters coming from command line
    if sys.argv[1] == 'gpu':
        gpu = True;
    else:
        gpu = False;
    method = sys.argv[2];
    N_x = int(sys.argv[3]);
    N_y = int(sys.argv[4]);
    hidden_ratio = int(sys.argv[5]);
    mixing_ratio = int(sys.argv[6]);
    Jx = float(sys.argv[7]);
    Jy = float(sys.argv[8]);
    Jz = float(sys.argv[9]);
    field = float(sys.argv[10]);
    gamma = float(sys.argv[11]);
    iteration = int(sys.argv[12]);
    MC_sweep = int(sys.argv[13]);
    thermal_rate = float(sys.argv[14]);
    learning_rate = float(sys.argv[15]);
    lru_size = int(sys.argv[16]);
    drawn = int(sys.argv[17]);
    restart = sys.argv[18];
    path = sys.argv[19];
    #######################################################################
               
    # Create Neural-Network Quantum State
    nqs = NQS(node_visible = N_x*N_y, node_hidden = hidden_ratio*N_x*N_y, node_mixing = mixing_ratio*N_x*N_y);
#     nqs.reInit(node_visible = N_x*N_y, node_hidden = hidden_ratio*N_x*N_y,node_mix = mixing_ratio*N_x*N_y);
    # If we want to start from previous state, or want to do padding
    # For padding it is assumed that there are less added nodes than the old problem
    if restart != 'none':
        nqs.restart(restart);
          
    # Create Lindbladian;
#     lindbladian = Heisenberg_2D_XYZ( N_x, N_y, Jx, Jy, Jz, gamma, field)
    lindbladian = Rydberg_2D_open(N_x, N_y, Jz, gamma, field);
#     lindbladian = Transverse_Ising_2D(N_x, N_y, Jz, gamma, field);
#     lindbladian = Heisenberg_Carleo(N_x, Jx, Jz, gamma, field);
    lindbladian.InitMoveDict();
                 
    # Set up the parameters for Metropolis-Hastings
    MH_params = {'sweep_MC': MC_sweep, 'thermal_rate': thermal_rate};
      
    # Set up LRU for efficient hashing
    #change it to have the same size as the MH sampling at each node!
    if method == 'ldag_y':
        lru = LRU(int(MC_sweep/MPI.COMM_WORLD.size));     
    else:
        lru = LRU(lru_size);  
     
    # Set up Metropolis Hastings
    if method == 'lind_eu' or method == 'lind_runge':
        mh = MH_lind(nqs, lindbladian, lru, drawn, gpu);
    elif method == 'ldag_full':
        mh = MH_ldag_full(nqs, lindbladian, lru, drawn, gpu);
    else:   
        mh = MH_ldag_y(nqs, lindbladian, lru, drawn, gpu);
       
    # Set up IO
    io = IO();
         
    # Initialize SR optimizer
    if method == 'lind_runge':
        sr = SR_Runge(nqs, lindbladian, mh, io, path, learning_rate, MH_params, MPI.COMM_WORLD, method, lru_size, drawn, gpu);        
    else:
        sr = SR(nqs, lindbladian, mh, io, path, learning_rate, MH_params, MPI.COMM_WORLD, method, lru_size, drawn, gpu);        
      
    # Run the optimization
    sr.sd_run(iteration, MPI.SUM);

#     nqs.reInit(N_x*N_y,hidden_ratio*N_x*N_y,mixing_ratio*N_x*N_y)
#     matTest = MatlabTester(N_x*N_y, N_x*N_y, hidden_ratio*N_x*N_y,mixing_ratio*N_x*N_y, nqs, lindbladian);
# #     matTest.Test(method, lru, drawn);
#     matTest.SR_with_base(method, lru, drawn, 1, io, learning_rate);    
     
    now = time.time();
    if MPI.COMM_WORLD.rank == 0:
        print('Time:', now-then);

          
main()


