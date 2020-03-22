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
 # MatlabTester.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

# Class to crosscheck with Matlab results
# Check density matrix elem calculation

# Absolutely NOT optimized. Runs once


#    Put THIS into MAIN for running the tests

#    nqs.reInit(node_visible = 2*N_x*N_y, node_hidden = hidden_ratio*N_x*N_y,node_mix = hidden_mixing*N_x*N_y);
#     ##########################################################
#     # Test
#     ##########################################################
#     matTest = MatlabTester(N_x*N_y, N_x*N_y, hidden_ratio*N_x*N_y,hidden_mixing*N_x*N_y, nqs, lindbladian);
#     matTest.Test();

import random
import numpy as np
from scipy.sparse.linalg import lsqr


from Classes import MH_ldag_y, MH_ldag_full, MH_lind, NQS


class MatlabTester(object):

    def __init__(self, N_site, node_vis, node_hid, node_mixing, nqs, lindbladian):
        
        self.N_site = N_site;
        self.node_vis = node_vis;
        self.node_hid = node_hid;
        self.node_mixing = node_mixing;
        self.nqs = nqs;
        self.lindbladian = lindbladian;
        
    def SR_with_base(self, method, lru, drawn, batch_nbr, io, learning_rate):
        
         # Initialize full basis
        basis, decimal = self.InitBasis();
        
        # Initialize sample
        # Initialize sample
        if method == 'lind_eu' or method == 'lind_runge':
            mh = MH_lind(self.nqs, self.lindbladian, lru, drawn,False);
        elif method == 'ldag_full':
            mh = MH_ldag_full(self.nqs, self.lindbladian, lru, drawn,False);
        else:
            mh = MH_ldag_y(self.nqs, self.lindbladian, lru, drawn, False);
            
        mh.Initialize(0.2*10000,10000)
        
        for batch in range(batch_nbr):
            
        # Calculate <rho|L|rho>
            val = 0;
            LOk = np.zeros((self.nqs.paramNbr,1));
            Okk = np.zeros((self.nqs.paramNbr,self.nqs.paramNbr));
            denom = 0;
            for i in range(2**self.N_site):
                for j in range(2**self.N_site):

                    self.nqs.lookup_init(basis[i], basis[j]);
                    mh.rho = self.nqs.rho_elm(basis[i], basis[j]);
                    mh.S_i = basis[i];
                    mh.S_j = basis[j];

 
                    local_rho = mh.local_Lrho();
                    devs = np.conj(mh.nqs.nqs_derivative(mh.S_i, mh.S_j));
                    LOk = np.add(LOk, (local_rho*devs + np.conj(local_rho)*np.conj(devs))*(np.conj(mh.rho)*mh.rho));
                    Okk = np.add(Okk, (np.dot(devs, np.conj(devs.T)) + np.dot(devs, np.conj(devs.T)).T)*(np.conj(mh.rho)*mh.rho));


                    val += local_rho*(np.conj(mh.rho)*mh.rho);
                    denom += np.conj(mh.rho)*mh.rho;

            
            Okk = Okk/denom;
            LOk = LOk/denom;
            val = val/denom;
            
            print(LOk)
            
            sol = lsqr(Okk, LOk, atol=1e-09, btol=1e-09);
            param_update = np.real(sol[0].reshape(self.nqs.paramNbr,1));
            
            self.nqs.nqs_update(learning_rate*param_update);
                
            print('Estimated <rho|L|rho> after batch nbr {}:    {}  '.format(batch+self.nqs.it_start, val));
                
            io.Output_Param(self.nqs);
            io.Output_Estimator(batch+self.nqs.it_start, val, 0.0);
        
        
    
    def Test(self,  method, lru, drawn):
        
        # Initialize full basis
        basis, decimal = self.InitBasis();
        
        # Initialize sample
        # Initialize sample
        if method == 'lind_eu' or method == 'lind_runge':
            mh = MH_lind(self.nqs, self.lindbladian, lru, drawn,False);
        elif method == 'ldag_full':
            mh = MH_ldag_full(self.nqs, self.lindbladian, lru, drawn,False);
        else:
            mh = MH_ldag_y(self.nqs, self.lindbladian, lru, drawn, False);
            
        mh.Initialize(0.2*10000,10000)
                
        # Reinitialize network values
        #self.nqs.reInit(self.node_vis, self.node_hid, self.node_mixing);
        
        # Calculate <rho|L|rho>
        val = 0;
        Ok = np.zeros((self.nqs.paramNbr,1));
        Okk = np.zeros((self.nqs.paramNbr,self.nqs.paramNbr));
        denom = 0;
        for i in range(2**self.N_site):
            for j in range(2**self.N_site):
#         for i in range(1):
#             for j in range(1):
                       
                self.nqs.lookup_init(basis[i], basis[j]);
                mh.rho = self.nqs.rho_elm(basis[i], basis[j]);
                mh.S_i = basis[i];
                mh.S_j = basis[j];

 
                local_rho = mh.local_Lrho();
                devs = np.conj(mh.nqs.nqs_derivative(mh.S_i, mh.S_j));
                Ok = np.add(Ok, devs*(np.conj(mh.rho)*mh.rho));
                Okk = np.add(Okk, np.dot(devs, np.conj(devs.T)) + np.dot(devs, np.conj(devs.T)).T*(np.conj(mh.rho)*mh.rho));
                
#                 print(local_rho)

                val += local_rho*(np.conj(mh.rho)*mh.rho);
                denom += np.conj(mh.rho)*mh.rho;
        
#         print(val/denom)        
        print(Okk/denom)
#         print(Ok/denom)
        
        
    
    def InitBasis(self):

        basis = []; 
        decimal = []; 
        for i in range(2**self.N_site):
            form = '{0:0' + str(self.N_site) + 'b}';
            bas = form.format(i);
            decimal.append(i);
            mm = [];
            for j in range(self.N_site):
                if int(bas[j]) == 0:
                    mm.append(-1);
                else:
                    mm.append(+1);
            mm = np.asarray(mm).reshape(self.N_site,1);
            basis.append(mm);
                
        return basis, decimal;




