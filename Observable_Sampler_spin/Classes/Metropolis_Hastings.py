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
import random

class Metropolis_Hastings(object):
    
    def __init__(self, nqs, lind, thermal, sweeps, obs):
        self.nqs = nqs;
        self.lind = lind;
        self.thermal = thermal;
        self.sweeps = sweeps;  
        self.obs = obs;
        
    def initState(self):
        (self.S_i, self.S_j) = self.lind.init_rnd_state();   
        self.nqs.lookup_init(self.S_i, self.S_j);   
        self.rho = self.nqs.rho_elm(self.S_i, self.S_j);
        self.move_rate = 0; 


    def thermalization(self, zeroField):
        print('MC thermalization with steps:', int(self.thermal*self.sweeps));
#         [self.move(zeroField) for _ in range(int(self.thermal*self.sweeps))];
        while self.move_rate < int(self.thermal*self.sweeps):
            self.move(zeroField);   
        print('Thermalization finished');
   
   
    def sampling(self, zeroField):
        
#         for _ in range(self.sweeps):
        self.move_rate = 0;
        it=0;
        while self.move_rate < self.sweeps:
            
            self.move(zeroField);           
            self.obs.calculate_local(self.S_i, self.S_j, it, np.conj(self.rho));
            it += 1;
             
        self.obs.Divide(it);
        self.obs.Accumulate_global();
        print("Sampling finished");
        print("      Sigma Z:  {}  +/-   {}".format((1/self.obs.nsite)*np.squeeze(np.sum(self.obs.sz)), (1/self.obs.nsite)*np.sqrt(np.sum(self.obs.var_sz**2))));
        print("      Sigma X:  {}  +/-   {}".format((1/self.obs.nsite)*np.squeeze(np.sum(self.obs.sx)), (1/self.obs.nsite)*np.sqrt(np.sum(self.obs.var_sx**2))));
        print("      Sigma Y:  {}  +/-   {}".format((1/self.obs.nsite)*np.squeeze(np.sum(self.obs.sy)), (1/self.obs.nsite)*np.sqrt(np.sum(self.obs.var_sy**2))));
    

    def move(self, zeroField):

        #generate a new state
        # in case of spin systems it's only represented by the flipped spin
        # otherwise it can be something different, depening on the representation chosen
#         if zeroField:
#             (flip_i, flip_j, transition_ratio) = self.lind.generate_state_ZERO(self.S_i, self.S_j);
#         else:
        (flip_i, flip_j) = self.lind.generate_state(self.S_i, self.S_j);
        
        
        #calculate acceptance probability
        ampl_ratio = self.nqs.amplitude_ratio(self.S_i, self.S_j, flip_i, flip_j);
        acc_prob = np.square(np.abs(ampl_ratio));

        
        #test if we accept the move, and update in lookup and current state
        if acc_prob > random.random() and acc_prob != 1: 
            self.nqs.lookup_update(self.S_i, self.S_j, flip_i, flip_j); 
            self.lind.update_state(self.S_i, self.S_j, flip_i, flip_j);
            self.rho = self.nqs.rho_elm(self.S_i, self.S_j);
            self.move_rate += 1;
            
            
            
            
            
    
    
