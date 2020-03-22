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
 # Rydberg_1D.py
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


class Rydberg_1D(object):
        
    def __init__(self, X_site, Jz, gamma, field):
        self.N_site = X_site;
        self.X_site = X_site;
        self.Y_site = 1;
        self.Jx = '-';
        self.Jy = '-';
        self.Jz = Jz;
        self.field = field;
        self.gamma = gamma;
        self.i = 1j;
        self.model_name = 'Rydberg_1D';
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
    
        
    def generate_state(self, S_i, S_j):     

        action = self.moveGen[self.rand.randint(0,5)];
             
        position = self.rand.randint(0, self.N_site);
        (flip_i, flip_j) = action(position, S_i, S_j); 
                   
        return flip_i, flip_j;  
    
    
       # calculates <x|L|rho> for a given state S,S'
    # returns with:
    #    - lindbladian_elements: list, the constants coming from acting <x|L
    #    - lindbladian_states: 2D array of list of pairs of integers, indicating which spins were flipped in the transition
    #           - [0]: in i
    #           - [1]: in j in case of <ij|
    #
    # WARNING: THIS IS VALID FOR 1D SYSTEM
    def left_on_L(self, S_i, S_j):
            
        #the diagonal terms are all stored on the 1st element
        lind_elm = [0];
        lind_states = [[None,None]];
        
        # the for loop considers all the terms in the Lindbladian expression
        # A, B, C, D, E as in the notes
        for col in range(self.X_site): #changed here
            
            pos = col;
            right_col = (self.X_site + (col + 1)%self.X_site)%self.X_site;
            right_neighbour = right_col;
             
            ######################################################################
            # HAMILTONIAN PART: A & B
            ######################################################################
             
            # Diagonal term from A
            lind_elm[0] += 0.25*self.Jz*(-self.i);
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_j[pos];
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_j[right_neighbour];
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_j[pos]*S_j[right_neighbour];
                 
            # Off-diagonal term from external field Sx
            if self.field != 0:
                lind_states.append([ None, [ pos ] ]);
                lind_elm.append( -self.i * self.field);   
                   
            ######################################################################
            ######################################################################
                  
            # Diagonal term from B
            lind_elm[0] += 0.25*self.Jz*self.i;
            lind_elm[0] += 0.25*self.Jz*self.i*S_i[pos];
            lind_elm[0] += 0.25*self.Jz*self.i*S_i[right_neighbour];
            lind_elm[0] += 0.25*self.Jz*self.i*S_i[pos]*S_i[right_neighbour];
                           
            # Off-diagonal term from external field Sx
            if self.field != 0:
                lind_states.append([[ pos ],None]);
                lind_elm.append( self.i *self.field );
            
            ######################################################################
            ######################################################################
                
            #Term C - D - E are in total diagonal and have only the following terms
            lind_elm[0] += -0.25*self.gamma;
            lind_elm[0] += 0.25*self.gamma*S_i[pos]*S_j[pos];
        
        
        return lind_elm, lind_states;
    
    
    def left_on_L_dag(self, S_i, S_j):
            
        #the diagonal terms are all stored on the 1st element
        lind_elm = [0];
        lind_states = [[None,None]];
                        
        # the for loop considers all the terms in the Lindbladian expression
        # A, B, C, D, E as in the notes
        for col in range(self.X_site):
            
            pos = col;
            right_col = (self.X_site + (col + 1)%self.X_site)%self.X_site;
            right_neighbour = right_col;
             
            ######################################################################
            # HAMILTONIAN PART: A & B
            ######################################################################
             
            # Diagonal term from A
            lind_elm[0] += 0.25*self.Jz*self.i;
            lind_elm[0] += 0.25*self.Jz*self.i*S_j[pos];
            lind_elm[0] += 0.25*self.Jz*self.i*S_j[right_neighbour];
            lind_elm[0] += 0.25*self.Jz*self.i*S_j[pos]*S_j[right_neighbour];

            # Off-diagonal term from external field Sx
            if self.field != 0:
                lind_states.append([ None, [ pos ] ]);
                lind_elm.append( self.i * self.field);
   
            ######################################################################
            ######################################################################
                  
            # Diagonal term from B
            lind_elm[0] += 0.25*self.Jz*(-self.i);
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_i[pos];
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_i[right_neighbour];
            lind_elm[0] += 0.25*self.Jz*(-self.i)*S_i[pos]*S_i[right_neighbour];

            # Off-diagonal term from external field Sx
            if self.field != 0:
                lind_states.append([[ pos ],None]);
                lind_elm.append( -self.i *self.field );
                    
            ######################################################################
            ######################################################################
                
            #Term C - D - E are in total diagonal and have only the following terms
            lind_elm[0] += -0.25*self.gamma;
            lind_elm[0] += 0.25*self.gamma*S_i[pos]*S_j[pos];
        
        
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

        
    
    
    
