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

# Implementation of the global NQS ansatz for the density matrix

# WARNING -- THIS IMPLEMENTATION WORKS FOR SPIN SYSTEMS!!!

import numpy as np
import re

class NQS(object):
    
    # Initiate the NQS object with given nbr of hidden and visible neurons
    # Variables:
    #      - complex: a, W, b, X
    #      - real: b_p
    def __init__(self, node_visible=10, node_hidden=10, node_mixing = 10, init_weight=0.000001):

        # Define the imaginary unit
        self.i = 1j;
        
        self.it_start = 0;
        print()
        # Assign random numbers for the biases and the weights
        a_re = init_weight*(np.random.rand(node_visible,1)-0.5);
        a_im = init_weight*(np.random.rand(node_visible,1)-0.5);
#         a_re = -1*np.ones((node_visible,1));
#         a_im = init_weight*(np.random.rand(node_visible,1)-0.5);
        
        b_re = init_weight*(np.random.rand(node_hidden,1)-0.5);
        b_im = init_weight*(np.random.rand(node_hidden,1)-0.5);
        X_re = init_weight*(np.random.rand(node_hidden,node_visible)-0.5);
        X_im = init_weight*(np.random.rand(node_hidden,node_visible)-0.5);
        W_re = init_weight*(np.random.rand(node_mixing,node_visible)-0.5);
        W_im = init_weight*(np.random.rand(node_mixing,node_visible)-0.5);
        
        self.bp = init_weight*(np.random.rand(node_mixing,1)-0.5);
        self.a = a_re + 1j*a_im;
        self.b = b_re + 1j*b_im;
        self.X = X_re + 1j*X_im;
        self.W = W_re + 1j*W_im;
                
        # Define the rest (theta will be b + W * input)
        self.node_visible = node_visible;
        self.node_hidden = node_hidden;
        self.node_mixing = node_mixing;
        self.paramNbr = (node_visible + node_hidden + node_visible*(node_mixing + node_hidden))*2 + node_mixing;
             
        # Support variables for parameter distribution
        self.a_re_end = node_visible;
        self.a_im_end = 2*node_visible;
        self.b_re_end = 2*node_visible + node_hidden;
        self.b_im_end = (node_visible + node_hidden) * 2;
        self.X_re_end = 2*(node_visible + node_hidden) + node_visible*node_hidden;  
        self.X_im_end = 2*(node_visible + node_hidden + node_visible*node_hidden);   
        self.bp_end = 2*(node_visible + node_hidden + node_visible*node_hidden) + node_mixing;
        self.W_re_end = 2*(node_visible + node_hidden + node_visible*node_hidden) + node_mixing*node_visible + node_mixing; 
    
    # if we want to restart from previous state, or do padding
    
    
    def restart(self, path):

        file = open(path,'r');
        data = file.read();
        data = re.split(' |\n',data);
        data = list(filter(None,data));
    
        # Visible node - Hidden node - Mixing node
        self.it_start = int(data[0]);
        init = [int(i) for i in data[2:5]];
        pad = [int(i) for i in data[6:9]];
        pars = [float(i) for i in data[9:]];
                
        # if we start from previous state
        if init == pad:
            a_re = np.asarray(pars[:self.a_re_end]).reshape(self.node_visible, 1);
            a_im = np.asarray(pars[self.a_re_end : self.a_im_end]).reshape(self.node_visible, 1);
            b_re = np.asarray(pars[self.a_im_end : self.b_re_end]).reshape(self.node_hidden, 1);
            b_im = np.asarray(pars[self.b_re_end : self.b_im_end]).reshape(self.node_hidden, 1);
            X_re = np.asarray(pars[self.b_im_end : self.X_re_end]).reshape(self.node_hidden,self.node_visible);
            X_im = np.asarray(pars[self.X_re_end : self.X_im_end]).reshape(self.node_hidden,self.node_visible);
            W_re = np.asarray(pars[self.bp_end : self.W_re_end]).reshape(self.node_mixing,self.node_visible);
            W_im = np.asarray(pars[self.W_re_end : ]).reshape(self.node_mixing,self.node_visible);
        
            self.a = a_re + 1j*a_im;
            self.b = b_re + 1j*b_im;
            self.X = X_re + 1j*X_im;
            self.W = W_re + 1j*W_im;
            self.bp = np.asarray(pars[self.X_im_end : self.bp_end]).reshape(self.node_mixing, 1);
           
        # if we do padding
        else:
            # load everything with old parameter number,     
            # then start copying it the missing times
            a_re = np.asarray(pars[:init[0]]).reshape(init[0], 1);
            a_im = np.asarray(pars[init[0] : 2*init[0]]).reshape(init[0], 1);
            b_re = np.asarray(pars[2*init[0] : 2*init[0] + init[1]]).reshape(init[1], 1);
            b_im = np.asarray(pars[2*init[0] + init[1] : (init[0] + init[1]) * 2]).reshape(init[1], 1);
            X_re = np.asarray(pars[(init[0] + init[1]) * 2 : 2*(init[0] + init[1]) + init[0]*init[1]]).reshape(init[1],init[0]);
            X_im = np.asarray(pars[2*(init[0] + init[1]) + init[0]*init[1] : 2*(init[0] + init[1] + init[0]*init[1])]).reshape(init[1],init[0]);
            W_re = np.asarray(pars[2*(init[0] + init[1] + init[0]*init[1]) + init[2] : 2*(init[0] + init[1] + init[0]*init[1]) + init[2]*init[0] + init[2]]).reshape(init[2],init[0]);
            W_im = np.asarray(pars[2*(init[0] + init[1] + init[0]*init[1]) + init[2]*init[0] + init[2] : ]).reshape(init[2],init[0]);
        
            a = a_re + 1j*a_im;
            b = b_re + 1j*b_im;
            X = X_re + 1j*X_im;
            W = W_re + 1j*W_im;
            bp = np.asarray(pars[2*(init[0] + init[1] + init[0]*init[1]) : 2*(init[0] + init[1] + init[0]*init[1]) + init[2]]).reshape(init[2], 1);
            
            # a
            times_vis = int(np.floor(pad[0]/init[0]));
            res_vis = pad[0]%init[0];
            self.a = np.concatenate([np.tile(a, (times_vis,1)), a[: res_vis]]);
            #################################################xx
            # CHANGE
            # copy in physical direction, 10-5 in hidden node direction
            #################################################xx
            # b
            times_hid = int(np.floor(pad[1]/init[1]));
            self.b = np.concatenate([b, 0.00001*np.ones((pad[1]-init[1], 1))]);

            # bp
            times_mix = int(np.floor(pad[2]/init[2]));
            self.bp = np.concatenate([bp, 0.00001*np.ones((pad[2]-init[2], 1))]);
            
            # X
            X_upper = np.concatenate([np.tile(X, (1,times_vis)), X[:, 0:res_vis]], axis=1);
            self.X = np.concatenate([X_upper, 0.00001*np.ones((pad[1]-init[1], pad[0]))], axis=0);
            
            # W
            W_upper = np.concatenate([np.tile(W, (1,times_vis)), W[:, 0:res_vis]], axis=1);
            self.W = np.concatenate([W_upper, 0.00001*np.ones((pad[2]-init[2], pad[0]))], axis=0);
                         
    
    # Update the parameters of the network
    def nqs_update(self, pars):

        a_re = np.asarray(pars[:self.a_re_end]).reshape(self.node_visible, 1);
        a_im = np.asarray(pars[self.a_re_end : self.a_im_end]).reshape(self.node_visible, 1);
        b_re = np.asarray(pars[self.a_im_end : self.b_re_end]).reshape(self.node_hidden, 1);
        b_im = np.asarray(pars[self.b_re_end : self.b_im_end]).reshape(self.node_hidden, 1);
        X_re = np.asarray(pars[self.b_im_end : self.X_re_end]).reshape(self.node_hidden,self.node_visible);
        X_im = np.asarray(pars[self.X_re_end : self.X_im_end]).reshape(self.node_hidden,self.node_visible);
        W_re = np.asarray(pars[self.bp_end : self.W_re_end]).reshape(self.node_mixing,self.node_visible);
        W_im = np.asarray(pars[self.W_re_end : ]).reshape(self.node_mixing,self.node_visible);
        
        self.a += a_re + 1j*a_im;
        self.b += b_re + 1j*b_im;
        self.X += X_re + 1j*X_im;
        self.W += W_re + 1j*W_im;
        self.bp += np.asarray(pars[self.X_im_end : self.bp_end]).reshape(self.node_mixing, 1);
    
    def parameter_set(self, pars):
        self.a = pars[:self.node_visible];
        self.b = pars[self.node_visible : self.node_visible+self.node_hidden];
        self.X = pars[self.node_visible+self.node_hidden : self.node_visible+self.node_hidden + self.node_visible*self.node_hidden].reshape(self.node_hidden,self.node_visible);
        self.bp = pars[self.node_visible+self.node_hidden + self.node_visible*self.node_hidden : self.node_visible+self.node_hidden + self.node_visible*self.node_hidden + self.node_mixing];
        self.W = pars[self.node_visible+self.node_hidden + self.node_visible*self.node_hidden + self.node_mixing :].reshape(self.node_mixing,self.node_visible);
        
    
    # Get all the parameters of the network as an array
    def parameter_get(self):
        return    np.concatenate([self.a,
                                self.b,
                                self.X.reshape(self.node_hidden*self.node_visible,1),
                                self.bp,
                                self.W.reshape(self.node_mixing*self.node_visible,1)
                                ]);
            
    
    # Initiate a look-up table to keep theta-s in memory
    def lookup_init(self, S_i, S_j):
      
        self.theta = self.bp + np.dot( self.W , S_i ) + np.dot( np.conj(self.W) , S_j);
        self.theta_Si = self.b + np.dot( self.X, S_i );
        self.theta_Sj = np.conj(self.b) + np.dot( np.conj(self.X), S_j);
  
    
    # Update the look-up table after a move suggested in the Metropolis-Hastings
    def lookup_update(self, S_i, S_j, flip_i, flip_j):
        
        # create vector for the changed posi
        if flip_i is None and flip_j is None :
            return;
        
        Mi = self.__Build_changed_state(S_i, flip_i);
        Mj = self.__Build_changed_state(S_j, flip_j);

        self.theta += np.dot( self.W, Mi) + np.dot( np.conj(self.W) , Mj);
        self.theta_Si += np.dot( self.X, Mi );
        self.theta_Sj += np.dot( np.conj(self.X), Mj);
    
    
    # Calculate the density matrix element for given input states
    def rho_elm(self, S_i, S_j):
        
        # exponential factors
        ex_i = np.exp(np.dot(self.a.T, S_i));
        ex_j = np.exp(np.dot( np.conj(self.a.T), S_j));
        
        # the cosh factors
        cosh_p = np.prod(np.cosh(self.theta), axis=0);
        cosh_i = np.prod(np.cosh(self.theta_Si), axis=0);
        cosh_j = np.prod(np.cosh(self.theta_Sj), axis=0);
            
        element = ex_i*ex_j*cosh_p*cosh_i*cosh_j;
 
        return np.squeeze(element);    
    

    # Calculate amplitude ratio for Metropolis-hastings acceptance criteria
    # Computationally it's better to compute the logarithmic ratio first
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
    
    
        # Calculate amplitude ratio for Metropolis-hastings acceptance criteria
    # Computationally it's better to compute the logarithmic ratio first
    def amplitude_ratio_double(self, S_i, S_j, flip_i_dag, flip_j_dag, flip_i, flip_j):
        
        if flip_i is None and flip_j is None and flip_i_dag is None and flip_j_dag is None:
            return 1;
         
        Mi = self.__Build_changed_state_double(S_i, flip_i_dag, flip_i);
        Mj = self.__Build_changed_state_double(S_j, flip_j_dag, flip_j);   
         
         
        log_ratio = np.dot(self.a.T, Mi) \
                  + np.dot( np.conj(self.a.T), Mj) \
                  + np.sum(np.log(np.cosh(self.theta + np.dot(self.W, Mi) + np.dot( np.conj(self.W) , Mj) )) - np.log(np.cosh(self.theta))) \
                  + np.sum(np.log(np.cosh(self.theta_Si + np.dot(self.X, Mi))) - np.log(np.cosh(self.theta_Si))) \
                  + np.sum(np.log(np.cosh(self.theta_Sj + np.dot( np.conj(self.X), Mj))) - np.log(np.cosh(self.theta_Sj)));
 
        return np.exp(log_ratio);   
    
    
    # Calculate derivatives of the NQS ansatz
    def nqs_derivative(self, S_i, S_j):
        
        i = 1j;
              
        D_a_re = S_i + S_j;
        D_a_im = 1j*(S_i-S_j);
        tan_i = np.tanh( self.theta_Si );
        tan_j = np.tanh( self.theta_Sj );  
        D_b_re = tan_i + tan_j;
        D_b_im = 1j*( tan_i - tan_j );
        D_X_re = (np.dot( tan_i, S_i.T ) + np.dot( tan_j, S_j.T )).reshape(self.node_hidden*self.node_visible,1);
        D_X_im = 1j*(np.dot( tan_i, S_i.T ) - np.dot( tan_j, S_j.T )).reshape(self.node_hidden*self.node_visible,1);
        D_bp = np.tanh(self.theta);
        D_W_re = np.dot( D_bp, D_a_re.T ).reshape(self.node_mixing*self.node_visible,1); 
        D_W_im = np.dot( D_bp, D_a_im.T ).reshape(self.node_mixing*self.node_visible,1);         
            
        return np.concatenate([D_a_re, D_a_im, D_b_re, D_b_im, D_X_re, D_X_im, D_bp, D_W_re, D_W_im]);
        
    
    # Function to produce M vectors, depending on the flipped spins
    @staticmethod
    def __Build_changed_state(S, flip):   
        
        M = np.zeros((len(S),1));
        
        if flip is not None:
            F = -2*S[flip];
            M.flat[flip] = F;
        
        return M;
    
    # Function to produce M vectors, depending on the flipped spins
    @staticmethod
    def __Build_changed_state_double(S, flip_dag, flip):   
        
        M_dag = np.zeros((len(S),1));
        M = np.zeros((len(S),1));
        if flip_dag is not None:
            F = -2*S[flip_dag];
            M_dag.flat[flip_dag] = F;
        if flip is not None:
            F = -2*S[flip];
            M.flat[flip] = F;
            
        return M_dag.astype(int)^M.astype(int);
    
    
    #########################################################################################
    #   Functions to test with MatLab results
    #########################################################################################
    
    # Reinitialization with given values
    def reInit(self, node_visible=10, node_hidden=10, node_mix = 10):
        
        i = 1j;
        
        a = [];
        for n in range(2*node_visible):
            a.append(-0.05);             
        self.a = np.asarray(a[:node_visible]) + i*np.asarray(a[node_visible:]);
               
        b = [];
        for n in range(2*node_hidden):
            b.append(-0.05);
        self.b = np.asarray(b[:node_hidden]) + i*np.asarray(b[node_hidden:]);
               
        W = [];           
        for n in range(2*node_mix*node_visible):
            W.append(-0.05);
        self.W = np.asarray(W[:node_mix*node_visible]) + i*np.asarray(W[node_mix*node_visible:]); 
        
        X = [];           
        for n in range(2*node_hidden*node_visible):
            X.append(-0.05);
        self.X = np.asarray(X[:node_hidden*node_visible]) + i*np.asarray(X[node_hidden*node_visible:]);  
        
        bp = [];
        for n in range(node_mix):
            bp.append(-0.05);
        self.bp = np.asarray(bp[:]);         
 
        self.a = np.asarray(self.a).reshape(node_visible,1);
        self.b = np.asarray(self.b).reshape(node_hidden,1);
        self.bp = np.asarray(self.bp).reshape(node_mix,1);
        self.W = np.asarray(self.W).reshape(node_mix,node_visible,order='F'); 
        self.X = np.asarray(self.X).reshape(node_visible,node_visible,order='F'); 

        
        
        
        
        
