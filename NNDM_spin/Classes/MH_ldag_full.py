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
 # MH_ldag_full.py
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

class MH_ldag_full(Metropolis_Hastings):
    
    def local_Lrho(self): 
        (elm_dagger, states_dagger) = self.lindbladian.left_on_L_dag(self.S_i, self.S_j);

        total_list=[];
        for (i, elm) in enumerate(elm_dagger):
            
            S_i_dag = self.S_i.copy();
            if states_dagger[i][0] is not None:
                S_i_dag[states_dagger[i][0]] *= -1;           
            S_j_dag = self.S_j.copy();
            if states_dagger[i][1] is not None:
                S_j_dag[states_dagger[i][1]] *= -1;  
            
            (lind_elm, lind_states) = self.lindbladian.left_on_L(S_i_dag, S_j_dag);  

            lind_list = [self.nqs.amplitude_ratio_double(self.S_i, self.S_j, states_dagger[i][0], states_dagger[i][1],lind_states[j][0], lind_states[j][1])*cof*elm 
                        for (j, cof) in enumerate(lind_elm)]

            total_list.append(np.squeeze(sum(lind_list)));
        
        return sum(total_list);
      
    
    def search_dictionary(self, key):
        
        if key in self.lru:
            local_L = self.lru[key];
        else:
            local_L = self.local_Lrho();
            self.lru[key] = local_L;
            
        return local_L;
        
            
            
            
            
            
            
