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
 # SR_Runge.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

# Implementing the stochastic reconfiguration algorithm

# Master: collecing data, solving SR, updating and writing out parameters
# Slave: initializing network, doing MC, sending collected data back

import numpy as np
try:
    import cupy as cp
    from cupy.testing import condition
    cuda_available = True;
except Exception as error:
    import numpy as cp
    cuda_available = False;
import time
from Classes.MinresQLP_gpu import MinresQLP_gpu
from Classes.MinresQLP_cpu import MinresQLP_cpu


class SR_Runge(object):
    
    # Initialize the object with optimization parameters
    def __init__(self, nqs, lindbladian, mh, io, path, learning_rate,params_sd, comm_world, method, lru_size, drawn, gpu):
               
        self.comm = comm_world;
        self.myrank = self.comm.rank;
        self.nqs = nqs;
        self.lindbladian = lindbladian;
        self.sweep_per_proc = int(params_sd['sweep_MC']/self.comm.size);
        self.sweep_MC = self.sweep_per_proc * self.comm.size;
        self.thermal_rate = params_sd['thermal_rate'];
        self.learning_rate = learning_rate;
        self.mh = mh;
        self.multiplier = +1;
        self.gpu = gpu and cuda_available;

        
        # Output simulation parameters
        if self.myrank == 0:
            self.io = io;
            self.io.Initialize(nqs,lindbladian, self.sweep_MC, self.thermal_rate, self.learning_rate, path, method, lru_size, drawn);
            self.io.Output_Param(self.nqs);


    # Run the MH sampler and calculate new network parameters batch_nbr times
    def sd_run(self, batch_nbr, oper):
        therm_state = None;
        pars = None;
                
        for batch in range(batch_nbr*2):
            
            batch_start = time.time();
                        
            self.mh.Initialize(self.thermal_rate*self.sweep_MC, self.sweep_per_proc);
     
            if self.myrank == 0:
                if not batch%2:
                    print('\nIteration nbr.:', batch/2);  
                    MAIN_PAR = self.nqs.parameter_get();         
                pars = self.nqs.parameter_get();
                self.mh.thermalization();                
                therm_state = (self.mh.S_i, self.mh.S_j);
                
            (therm_state, pars) = self.comm.bcast((therm_state, pars), root=0);
    
            if self.myrank != 0:
                self.nqs.parameter_set(pars);
                self.mh.update_state(therm_state);
                self.nqs.lookup_init(self.mh.S_i, self.mh.S_j);
             
            # order:
            # estimated = (Lrho, Ok, Okk, LOk, L_block, L_var)
            # need to boradcast one-by-one due to shape
           
            MH_start = time.time();
           
            if self.gpu:
                with cp.cuda.Device(self.myrank):
                    sampled = self.mh.sampling();
            else:
                sampled = self.mh.sampling();
            
            MH_stop = time.time();
            
            total_sweep = self.comm.reduce(sampled[0], op = oper, root = 0);
            Lrho = self.comm.reduce(sampled[1], op = oper, root = 0);
            Ok = self.comm.reduce(sampled[2], op = oper, root = 0);
            Okk = self.comm.reduce(sampled[3], op = oper, root = 0);
            LOk = self.comm.reduce(sampled[4], op = oper, root = 0);
            L_sq = self.comm.reduce(sampled[5], op = oper, root = 0);
            move_rate = self.comm.reduce(sampled[6], op = oper, root = 0);

            comm_done = time.time();

            if self.myrank == 0:
                Lrho = np.real(Lrho/total_sweep);
                Ok = np.real(Ok/total_sweep);
                Okk = np.real(Okk/total_sweep);
                LOk = np.real(LOk/total_sweep);
                L_sq = np.real(L_sq/self.comm.size);
                move_rate = move_rate/total_sweep;

                if self.gpu:
                    with cp.cuda.Device(self.myrank):
                        params_update, conditioned = self.stochastic_gradient(Lrho, Ok, Okk, LOk, batch); 
                else:
                    params_update, conditioned = self.stochastic_gradient(Lrho, Ok, Okk, LOk, batch); 
                
                SR_solve_done = time.time();
                
                
                if not batch%2:
                    self.nqs.nqs_update(self.learning_rate/2*params_update);
                else:
                    self.nqs.parameter_set(MAIN_PAR);
                    self.nqs.nqs_update(self.learning_rate*params_update);
                
                    print('Estimated <rho|L|rho> after batch nbr {}:    {}  +/-   {}, move {}, total {}. cond {}'.format(batch+self.nqs.it_start, Lrho, L_sq, move_rate, total_sweep,conditioned));
                    self.io.Output_Param(self.nqs);
                    self.io.Output_Estimator((batch-1+self.nqs.it_start)/2, Lrho, L_sq);
                    self.io.Output_Profiling(move_rate,MH_start-batch_start, MH_stop-MH_start, comm_done-MH_stop, SR_solve_done-comm_done);
          
            
    # Calculate new network parameters via SR
    def stochastic_gradient(self, lrho, Ok, Okk, LOk, it):
 
        S_kk = np.subtract(Okk, np.dot(Ok, np.conj(Ok.T)));
        F_k = self.multiplier * np.subtract(LOk, lrho*Ok);

#         S_kk_diag = np.zeros(S_kk.shape, dtype=complex);
#         row, col = np.diag_indices(S_kk.shape[0]);
#         S_kk_diag[row, col] = self.lambd(it, it_start=self.nqs.it_start) * np.diagonal(S_kk);
# 
#         S_kk_reg = np.add(S_kk, S_kk_diag);

        row, col = np.diag_indices(S_kk.shape[0]);
        S_kk[row, col] += self.lambd(it, it_start=self.nqs.it_start);
        
        rtol = 1e-11;
        maxit = 20000;
                 
        if self.gpu:       
            sol = MinresQLP_gpu(cp.array(S_kk), F_k, rtol, maxit);         
        else:
            sol = MinresQLP_cpu(S_kk, F_k, rtol, maxit);   
        return np.real(sol[0]), np.linalg.cond(S_kk)          

        
#         sol = np.linalg.solve(S_kk_reg, F_k);
#         return np.real(sol), np.linalg.cond(S_kk_reg)


    # lambda regularization from paper
    @staticmethod
    def lambd(it, lambd0=100, b=0.98, lambdMin=1e-4, it_start=0):
#         return max(lambd0 * (b ** (it + it_start)), lambdMin);
        return lambdMin;
    
        
        
        
        
        
        
        
