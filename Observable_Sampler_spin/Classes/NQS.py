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
 # NQS.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

import re
import numpy as np


class NQS(object):
    
    def __init__(self,node_visible,  node_hidden, node_mixing):
        self.node_visible = node_visible;
        self.node_hidden = node_hidden;
        self.node_mixing = node_mixing;
        
        # Define the rest (theta will be b + W * input)
        self.paramNbr = (self.node_visible + self.node_hidden + self.node_visible*(self.node_mixing + self.node_hidden))*2 + self.node_mixing;
#         self.paramNbr = (self.node_visible + self.node_hidden + self.node_visible*(self.node_mixing + self.node_hidden))*2;


        # Support variables for parameter distribution
        self.a_re_end = self.node_visible;
        self.a_im_end = 2*self.node_visible;
        self.b_re_end = 2*self.node_visible + self.node_hidden;
        self.b_im_end = (self.node_visible + self.node_hidden) * 2;
        self.X_re_end = 2*(self.node_visible + self.node_hidden) + self.node_visible*self.node_hidden;  
        self.X_im_end = 2*(self.node_visible + self.node_hidden + self.node_visible*self.node_hidden);   
        self.bp_end = 2*(self.node_visible + self.node_hidden + self.node_visible*self.node_hidden) + self.node_mixing;
        self.W_re_end = 2*(self.node_visible + self.node_hidden + self.node_visible*self.node_hidden) + self.node_mixing*self.node_visible + self.node_mixing; 
#         self.W_re_end = 2*(self.node_visible + self.node_hidden + self.node_visible*self.node_hidden) + self.node_mixing*self.node_visible; 
        
        

    def param_set(self, line):
        
        line = re.split(' |\n', line);
        line = list(filter(None,line));
        pars = [float(i) for i in line];

        a_re = np.asarray(pars[:self.a_re_end]).reshape(self.node_visible, 1);
        a_im = np.asarray(pars[self.a_re_end : self.a_im_end]).reshape(self.node_visible, 1);
        b_re = np.asarray(pars[self.a_im_end : self.b_re_end]).reshape(self.node_hidden, 1);
        b_im = np.asarray(pars[self.b_re_end : self.b_im_end]).reshape(self.node_hidden, 1);
        X_re = np.asarray(pars[self.b_im_end : self.X_re_end]).reshape(self.node_hidden,self.node_visible);
        X_im = np.asarray(pars[self.X_re_end : self.X_im_end]).reshape(self.node_hidden,self.node_visible);
        W_re = np.asarray(pars[self.bp_end : self.W_re_end]).reshape(self.node_mixing,self.node_visible);
#         W_re = np.asarray(pars[self.X_im_end : self.W_re_end]).reshape(self.node_mixing,self.node_visible);
        W_im = np.asarray(pars[self.W_re_end : ]).reshape(self.node_mixing,self.node_visible);
        
        self.a = a_re + 1j*a_im;
        self.b = b_re + 1j*b_im;
        self.X = X_re + 1j*X_im;
        self.W = W_re + 1j*W_im;
        self.bp = np.asarray(pars[self.X_im_end : self.bp_end]).reshape(self.node_mixing, 1);
        
        
    # Initiate a look-up table to keep theta-s in memory
    def lookup_init(self, S_i, S_j):
      
        self.theta = self.bp + np.dot( self.W , S_i ) + np.dot( np.conj(self.W) , S_j);
#         self.theta = np.dot( self.W , S_i ) + np.dot( np.conj(self.W) , S_j);
        self.theta_Si = self.b + np.dot( self.X, S_i );
        self.theta_Sj = np.conj(self.b) + np.dot( np.conj(self.X), S_j);
        
    def lookup_update(self, S_i, S_j, flip_i, flip_j):
        
        # create vector for the changed posi
        if flip_i is None and flip_j is None :
            return;
        
        Mi = self.__Build_changed_state(S_i, flip_i);
        Mj = self.__Build_changed_state(S_j, flip_j);

        self.theta += np.dot( self.W, Mi) + np.dot( np.conj(self.W) , Mj);
        self.theta_Si += np.dot( self.X, Mi );
        self.theta_Sj += np.dot( np.conj(self.X), Mj);
        
        
    def amplitude_ratio(self, S_i, S_j, flip_i, flip_j):
        
        if flip_i is None and flip_j is None :
            return 1;
        
        Mi = self.__Build_changed_state(S_i, flip_i);
        Mj = self.__Build_changed_state(S_j, flip_j);      
         
        log_ratio = np.dot(self.a.T, Mi) \
                  + np.dot( np.conj(self.a.T), Mj) \
                  + np.sum(np.log(np.cosh(self.theta + np.dot(self.W, Mi) + np.dot( np.conj(self.W) , Mj) )) - np.log(np.cosh(self.theta))) \
                  + np.sum(np.log(np.cosh(self.theta_Si + np.dot(self.X, Mi))) - np.log(np.cosh(self.theta_Si))) \
                  + np.sum(np.log(np.cosh(self.theta_Sj + np.dot( np.conj(self.X), Mj))) - np.log(np.cosh(self.theta_Sj)));

        return np.exp(log_ratio);   
    

    def rho_elm(self, S_i, S_j):
        
        # exponential factors
        ex_i = np.exp(np.dot(self.a.T, S_i));
        ex_j = np.exp(np.dot( np.conj(self.a.T), S_j));
        
        # the cosh factors
        cosh_p = np.prod(np.cosh(self.bp + np.dot( self.W , S_i ) + np.dot( np.conj(self.W) , S_j)), axis=0);
#         cosh_p = np.prod(np.cosh(np.dot( self.W , S_i ) + np.dot( np.conj(self.W) , S_j)), axis=0);
        cosh_i = np.prod(np.cosh(self.b + np.dot( self.X, S_i )), axis=0);
        cosh_j = np.prod(np.cosh(np.conj(self.b) + np.dot( np.conj(self.X), S_j)), axis=0);

            
        element = ex_i*ex_j*cosh_p*cosh_i*cosh_j;
 
        return np.squeeze(element);  
        
   
    @staticmethod
    def __Build_changed_state(S, flip):   
        
        M = np.zeros((len(S),1));
        
        if flip is not None:
            F = -2*S[flip];
            M.flat[flip] = F;
        
        return M;
         
        
        
