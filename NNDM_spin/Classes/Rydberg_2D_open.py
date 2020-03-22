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
 # Rydberg_2D_open.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

import random
import numpy as np
from numpy import reshape


class Rydberg_2D_open(object):
        
    def __init__(self, X_site, Y_site, Jz, gamma, field):
        self.N_site = X_site*Y_site;
        self.X_site = X_site;
        self.Y_site = Y_site;
        self.Jx = '-';
        self.Jy = '-';
        self.Jz = Jz;
        self.field = field;
        self.gamma = gamma;
        self.i = 1j;
        self.model_name = 'Rydberg_2D';
        self.rand = np.random.RandomState();    
        
    
    def update_state(self, S_i, S_j, flip_i, flip_j):    
        if flip_i is not None:
            S_i.flat[flip_i] = -1*S_i[flip_i];
        if flip_j is not None:
            S_j.flat[flip_j] = -1*S_j[flip_j];


#     random initial state generation
    def init_rnd_state(self):
#         S_i = np.asarray([-1 for _ in range(self.N_site)]).reshape(self.N_site, 1);
#         S_j = np.asarray([-1 for _ in range(self.N_site)]).reshape(self.N_site, 1);
        S_i = np.asarray([random.randrange(-1, 2, 2) for _ in range(self.N_site)]).reshape(self.N_site, 1);
        S_j = np.asarray([random.randrange(-1, 2, 2) for _ in range(self.N_site)]).reshape(self.N_site, 1);
        return S_i, S_j     
    
        
#     def generate_state(self, S_i, S_j):     
# 
#         action = self.moveGen[self.rand.randint(0,5)];
#              
#         position = self.rand.randint(0, self.N_site);
#         (flip_i, flip_j) = action(position, S_i, S_j); 
#                    
#         return flip_i, flip_j;  
    
    def generate_state(self, S_i, S_j):     
        if 0.5 > random.random():
            flip_i = random.randint(0, self.N_site - 1);
            flip_j = None;
        else:
            flip_i = None;
            flip_j = random.randint(0, self.N_site - 1);
             
        return flip_i, flip_j;
    
    
    def left_on_L(self, S_i, S_j):
        
        lind_elm = [0];
        lind_states = [[None,None]];
        
        S_i = np.reshape(S_i, [self.Y_site,self.X_site]);
        # nearest neighbour on S_i
        lind_elm[0] += np.sum(self.i*0.25*self.Jz*np.multiply(1+S_i[:,:-1],1+S_i[:,1:]));
        lind_elm[0] += np.sum(self.i*0.25*self.Jz*np.multiply(1+S_i[:-1,:],1+S_i[1:,:]));
        
        S_j = np.reshape(S_j, [self.Y_site,self.X_site]);
        # nearest neighbour on S_j
        lind_elm[0] += np.sum(-self.i*0.25*self.Jz*np.multiply(1+S_j[:,:-1],1+S_j[:,1:]));
        lind_elm[0] += np.sum(-self.i*0.25*self.Jz*np.multiply(1+S_j[:-1,:],1+S_j[1:,:]));
        
        # field S_i, S_j
        if self.field != 0:
            lind_states.extend([[None,[i]] for i in range(self.N_site)]);
            lind_elm.extend([-self.i*self.field for _ in range(self.N_site)]);
            lind_states.extend([[[i],None] for i in range(self.N_site)]);
            lind_elm.extend([self.i*self.field for _ in range(self.N_site)]);
        
       
        # Dissipator
        S_i = np.reshape(S_i, [self.N_site,1]);
        S_j = np.reshape(S_j, [self.N_site,1]);
        lind_elm[0] += np.sum(-0.25*self.gamma*(1-np.multiply(S_i,S_j)));
        
        return lind_elm, lind_states;


    def left_on_L_dag(self, S_i, S_j):
        
        lind_elm = [0];
        lind_states = [[None,None]];
        
        S_i = np.reshape(S_i, [self.Y_site,self.X_site]);
        # nearest neighbour on S_i
        lind_elm[0] += np.sum(-self.i*0.25*self.Jz*np.multiply(1+S_i[:,:-1],1+S_i[:,1:]));
        lind_elm[0] += np.sum(-self.i*0.25*self.Jz*np.multiply(1+S_i[:-1,:],1+S_i[1:,:]));
        
        S_j = np.reshape(S_j, [self.Y_site,self.X_site]);
        # nearest neighbour on S_j
        lind_elm[0] += np.sum(self.i*0.25*self.Jz*np.multiply(1+S_j[:,:-1],1+S_j[:,1:]));
        lind_elm[0] += np.sum(self.i*0.25*self.Jz*np.multiply(1+S_j[:-1,:],1+S_j[1:,:]));
        
        # field S_i, S_j
        if self.field != 0:
            lind_states.extend([[None,[i]] for i in range(self.N_site)]);
            lind_elm.extend([self.i*self.field for _ in range(self.N_site)]);
            lind_states.extend([[[i],None] for i in range(self.N_site)]);
            lind_elm.extend([-self.i*self.field for _ in range(self.N_site)]);
               
        # Dissipator
        S_i = np.reshape(S_i, [self.N_site,1]);
        S_j = np.reshape(S_j, [self.N_site,1]);
        lind_elm[0] += np.sum(-0.25*self.gamma*(1-np.multiply(S_i,S_j)));
        
        return lind_elm, lind_states;
 
      
    
    def InitMoveDict(self):         
    #########################################################
    # Defining all the Linbladian move functions for the dictionary
    # Python does not have switch-case statements    
    #########################################################
        def hopping_column_i(pos, S_i, S_j):
            return [pos, self.CalcRightNN(pos,self.X_site,self.Y_site)], None;
    
#         def hopping_row_i(pos, S_i, S_j):
#             return [pos, self.CalcDownNN(pos,self.X_site,self.Y_site)], None;
    
        def hopping_column_j(pos, S_i, S_j):
            return None, [pos, self.CalcRightNN(pos,self.X_site,self.Y_site)];
    
#         def hopping_row_j(pos, S_i, S_j):
#             return None, [pos, self.CalcDownNN(pos,self.X_site,self.Y_site)];
        
        def external_i(pos, S_i, S_j):
            return pos, None;
        
        def external_j(pos, S_i, S_j):
            return None, pos;

        def dissipator(pos, S_i, S_j):
            if S_i[pos] == S_j[pos]: 
                return pos, pos;                            
            else:
                return None, None;
            
        def jumper(pos, S_i, S_j):
            pos_i = random.randrange(0, self.N_site);
            pos_j = random.randrange(0, self.N_site);
            
            return pos_i, pos_j;
                        
             
    #########################################################         
        self.moveGen = {
            0: hopping_column_i,
#             1: hopping_row_i,
            1: hopping_column_j,
#             3: hopping_row_j,
            2: dissipator,
            3: external_i,
            4: external_j,
            5: jumper
            }
        
    @staticmethod
    def CalcRightNN(pos, Nx, Ny):
        
        col = pos % Nx;
        row = np.floor((pos)/Nx)%Ny;       
        right_col = (Nx + (col + 1)%Nx)%Nx;
        right_neighbour = right_col + row*Nx;
        
        return int(right_neighbour);
    
    @staticmethod
    def CalcDownNN(pos, Nx, Ny):
        
        col = pos % Nx;
        row = np.floor((pos)/Nx)%Ny;       
        down_row = (Ny + (row + 1)%Ny)%Ny;
        down_neighbour = col + down_row*Nx;
        
        return int(down_neighbour);

        
    
    
    
