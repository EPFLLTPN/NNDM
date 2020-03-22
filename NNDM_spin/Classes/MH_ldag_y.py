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
 # MH_ldag_y.py
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
from Classes.Metropolis_Hastings import Metropolis_Hastings

class MH_ldag_y(Metropolis_Hastings):
    
    def local_Lrho(self): 
        (elm_dagger, states) = self.lindbladian.left_on_L_dag(self.S_i, self.S_j);
          
        N_x = len(elm_dagger);
        
        local_list = [];
        for d in range(self.draw):
            
            which = self.rand_state.randint(0, N_x);
                          
            S_i_y = self.S_i.copy();
            if states[which][0] is not None:
                S_i_y[states[which][0]] *= -1;           
            S_j_y = self.S_j.copy();
            if states[which][1] is not None:
                S_j_y[states[which][1]] *= -1; 
               
            (lind_elm, lind_states) = self.lindbladian.left_on_L(S_i_y, S_j_y); 
            lind_list = [self.nqs.amplitude_ratio_double(self.S_i, self.S_j, states[which][0], states[which][1],lind_states[i][0], lind_states[i][1])*elm*elm_dagger[which]
                    for (i, elm) in enumerate(lind_elm)]
            
            local_list.append(np.squeeze(sum(lind_list))*N_x);
           
        return local_list;
      
    
    def search_dictionary(self, key):
        
        curr_L = self.local_Lrho();
        if key in self.lru:
            (local_L, N_samp) = self.lru[key];
            (new_loc, new_step) = self.RunningAverage(local_L, N_samp, curr_L);
        else:
            (new_loc, new_step) = self.RunningAverage(0, 0, curr_L);   
        self.lru[key] = (new_loc, new_step);
        
        return new_loc;
        
        
    # performs a running average LOOP from many input updates
    @staticmethod
    def RunningAverage(avg, steps, new_data):
        for (i, d) in enumerate(new_data):
            avg += (d - avg)/(steps + i + 1);
        return avg, steps+len(new_data);
            
            
            
            
            
            
