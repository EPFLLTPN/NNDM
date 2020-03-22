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
 # Metropolis_Hastings.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #


import numpy as np
from audioop import lin2adpcm
try:
    import cupy as cp
    cuda_available = True;
except Exception as error:
    import numpy as cp
    cuda_available = False;
import heapq
import re
# WARNING
# some function definitons are blanc. You need to define them method specifically
# see in the beginning!!!

class Metropolis_Hastings(object):
    
    def local_Lrho(self):
        pass
    def search_dictionary(self):
        pass
    def RunningAverage(self):
        pass
    
    def __init__(self, nqs, lindbladian, lru, draw, gpu):
        self.nqs = nqs;
        self.lindbladian = lindbladian;
        self.lru = lru;
        self.draw = draw;
        self.gpu = gpu and cuda_available;
        self.largestDict = {};
        if not self.gpu:
            global cp;
            global np;
            cp = np;
    
    # not in __init__ because of module importing reasons
    def Initialize(self, thermal, sweeps):

        self.thermal = thermal;
        self.sweeps = sweeps;
        self.move_rate = 0;
                     
        self.reset_estimators();
        (self.S_i, self.S_j) = self.lindbladian.init_rnd_state();
        self.nqs.lookup_init(self.S_i, self.S_j);
        
        # inidvidual random generator
        # needed to get independent random seeds in parallel processing
        self.rand_state = np.random.RandomState();

    # reset estimator values
    def reset_estimators(self):
        self.L = 0.0;
        self.Ok = cp.zeros((self.nqs.paramNbr,1));
        self.Okk = cp.zeros((self.nqs.paramNbr,self.nqs.paramNbr));
        self.LOk = cp.zeros((self.nqs.paramNbr,1));  
        self.L_sq = 0.0;
        self.move_rate = 0;
               
    # updating the current state, used in parallelization
    def update_state(self, therm_state):
        self.S_i = therm_state[0];
        self.S_j = therm_state[1];                           
                
    # MC thermalization  
    def thermalization(self):
        
        while self.move_rate < int(self.thermal):
            self.move();        
#         (self.move() for _ in range(int(self.thermal)));

        
    # MC sampling
    def sampling(self):
                
        self.reset_estimators();
        
        # for running average
        L_run = 0.0;

        # Clear LRU
        self.lru.clear();

        # do it sweep times
        it=0;
        while self.move_rate < self.sweeps:
#         for i in range(self.sweeps):
        
            #make a move
            self.move();
            
            # calculate estimators and expectation values for SR
            key = str(self.S_i) + str(self.S_j);
            local_L = self.search_dictionary(key);
            
            devs = cp.conj(cp.asarray(self.nqs.nqs_derivative(self.S_i, self.S_j)));

            self.L += local_L;
            self.Ok = cp.add(self.Ok, devs);

            self.Okk = cp.add(self.Okk, cp.dot(devs, cp.conj(devs.T)) + cp.dot(devs, cp.conj(devs.T)).T);
#             self.Okk = cp.add(self.Okk, cp.dot(devs, cp.conj(devs.T)));

            local_L_gpu = cp.asarray(local_L);
            
            self.LOk = cp.add(self.LOk, local_L_gpu*devs + cp.conj(local_L_gpu)*cp.conj(devs));
#             self.LOk = cp.add(self.LOk, local_L_gpu*devs);


            # calculate batched running variance
            delta = np.real(local_L) - L_run;
            L_run += delta/(it+1);
            delta2 = np.real(local_L) - L_run;
            self.L_sq += delta*delta2;  
            it += 1;
            
        # Finalizing the blocked running variance calculation
        self.L_sq = np.sqrt(self.L_sq)/(it - 1);  
        
        if self.gpu:
            return it, self.L, cp.asnumpy(self.Ok), cp.asnumpy(self.Okk), cp.asnumpy(self.LOk), self.L_sq, self.move_rate, len(self.lru);
        else:
            return it, self.L, self.Ok, self.Okk, self.LOk, self.L_sq, self.move_rate, len(self.lru);

    # Making a MC move
    def move(self):

        (flip_i, flip_j) = self.lindbladian.generate_state(self.S_i, self.S_j);
         
        #calculate acceptance probability
        ampl_ratio = self.nqs.amplitude_ratio(self.S_i, self.S_j, flip_i, flip_j);
        acc_prob = np.square(np.abs(ampl_ratio));
        
        #test if we accept the move, and update in lookup and current state
        if acc_prob > self.rand_state.random_sample() and acc_prob != 1: 
            self.nqs.lookup_update(self.S_i, self.S_j, flip_i, flip_j);    
            self.lindbladian.update_state(self.S_i, self.S_j, flip_i, flip_j);
            self.move_rate += 1;   
  
    
    @staticmethod
    def change(S_i, S_j, flip_i, flip_j):    
        n1 = S_i.copy();
        n2 = S_j.copy();
        if flip_i is not None:
            n1.flat[flip_i] = -1*S_i[flip_i];
        if flip_j is not None:
            n2.flat[flip_j] = -1*S_j[flip_j];
            
        return n1, n2;

        
        
