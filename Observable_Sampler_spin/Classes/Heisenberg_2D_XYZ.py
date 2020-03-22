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
 # Heisenberg_2D_XYZ.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

# Implementation for the XYZ Heisenberg model on a 2-dimensional lattice

import random
import numpy as np


class Heisenberg_2D_XYZ(object):
        
    def __init__(self, X_site, Y_site, Jx, Jy, Jz, gamma, field):
        self.N_site = X_site*Y_site;
        self.X_site = X_site;
        self.Y_site = Y_site;
        self.Jx = Jx;
        self.Jy = Jy;
        self.Jz = Jz;
        self.field = field;
        self.gamma = gamma;
        self.i = 1j;
    
    def update_state(self, S_i, S_j, flip_i, flip_j):    
        if flip_i is not None:
            S_i.flat[flip_i] = -1*S_i[flip_i];
        if flip_j is not None:
            S_j.flat[flip_j] = -1*S_j[flip_j];

#     random initial state generation
    def init_rnd_state(self):
        S_i = np.asarray([random.randrange(-1, 2, 2) for _ in range(self.N_site)]).reshape(self.N_site, 1); 
        S_j = np.asarray([random.randrange(-1, 2, 2) for _ in range(self.N_site)]).reshape(self.N_site, 1);             
#         Start from all spin down
#         Avoid using random. The ground state is closed to all spin down -> faster thermalization
#        S_i = np.asarray([-1 for _ in range(self.N_site)]).reshape(self.N_site, 1);
#        S_j = np.asarray([-1 for _ in range(self.N_site)]).reshape(self.N_site, 1);
        return S_i, S_j     
    
        
    # Generate new state via flipping a spin in both i and j. going from rho_ij -> rho_lm
    #old way :
#         if 0.5 > random.random():
#             flip_i = random.randint(0, self.N_site - 1);
#             flip_j = None;
#         else:
#             flip_i = None;
#             flip_j = random.randint(0, self.N_site - 1);
#             
#         return flip_i, flip_j, 1;
    def generate_state(self, S_i, S_j):     
  
        action = self.moveGen[random.randrange(0,8)];
               
        position = random.randrange(0, self.N_site);
        (flip_i, flip_j, transition_ratio) = action(position, S_i, S_j); 
                     
        return flip_i, flip_j, transition_ratio;
    
    
    # Generate new state via flipping a spin in both i and j. going from rho_ij -> rho_lm
    # old way:
    #    flip_i = random.randint(0, self.N_site - 1);
    #    flip_j = random.randint(0, self.N_site - 1);
    # new way:
    # generate steps based on the Lindbladian connectivity
    # also randomly go back to diagonal
    def generate_state_ZERO(self, S_i, S_j): 
        
        action = self.moveGen[random.randrange(0,5)];
            
        position = random.randrange(0, self.N_site);
        (flip_i, flip_j, transition_ratio) = action(position, S_i, S_j); 
                  
        return flip_i, flip_j, transition_ratio;
    
    
    
    # calculates <x|L|rho> for a given state S,S'
    # returns with:
    #    - lindbladian_elements: list, the constants coming from acting <x|L
    #    - lindbladian_states: 2D array of list of pairs of integers, indicating which spins were flipped in the transition
    #           - [0]: in i
    #           - [1]: in j in case of <ij|
    #
    # WARNING: THIS IS VALID FOR 1D SYSTEM
    def find_lindbladian_elements(self, S_i, S_j):
            
        #the diagonal terms are all stored on the 1st element
        lind_elm = [0];
        lind_states = [[[],[]]];
        
        #Do we have field?
        if self.field == 0:
            external = False;
        else:
            external = True;
                        
        #Do we have double hopping?
        if self.Jx != self.Jy:
            double_hopping = True;
        else:
            double_hopping = False;
        
        # the for loop considers all the terms in the Lindbladian expression
        # A, B, C, D, E as in the notes
        for row in range(self.Y_site):
            for col in range(self.X_site):
            
                pos = col + row*self.X_site;

                right_col = (self.X_site + (col + 1)%self.X_site)%self.X_site;
                down_row = (self.Y_site + (row + 1)%self.Y_site)%self.Y_site;
                right_neighbour = right_col + row*self.X_site;
                down_neighbour = col + down_row*self.X_site;
             
                ######################################################################
                # HAMILTONIAN PART: A & B
                ######################################################################
                 
                # Diagonal term from A
                lind_elm[0] += self.Jz*(-self.i)*S_j[pos]*S_j[right_neighbour];
                lind_elm[0] += self.Jz*(-self.i)*S_j[pos]*S_j[down_neighbour];                
             
                # Off-diagonal term from A (hopping and double hopping) -> COL
                if S_j[pos] != S_j[right_neighbour]:
                    lind_elm.append((self.Jx + self.Jy)*(-self.i));
                    lind_states.append([ None, [pos, right_neighbour] ]);
                else:
                    if double_hopping:
                        lind_elm.append((self.Jx - self.Jy)*(-self.i));
                        lind_states.append([ None, [pos, right_neighbour] ]);
                    
                # Off-diagonal term from A (hopping and double hopping) -> ROW
                if S_j[pos] != S_j[down_neighbour]:
                    lind_elm.append((self.Jx + self.Jy)*(-self.i));
                    lind_states.append([ None, [pos, down_neighbour] ]);
                else:
                    if double_hopping:
                        lind_elm.append((self.Jx - self.Jy)*(-self.i));
                        lind_states.append([ None, [pos, down_neighbour] ]);
                 
                # Off-diagonal term from external field Sx
                if external:
                    lind_states.append([ None, [ pos ] ]);
                    lind_elm.append( -self.i * self.field);
   
                   
               
                ######################################################################
                ######################################################################
                      
                # Diagonal term from B
                lind_elm[0] += self.Jz*self.i*S_i[pos]*S_i[right_neighbour];
                lind_elm[0] += self.Jz*self.i*S_i[pos]*S_i[down_neighbour];
                      
                # Off-diagonal term from B (hopping and double hopping) -> COL
                if S_i[pos] != S_i[right_neighbour]:
                    lind_states.append([ [pos, right_neighbour], None ]);
                    lind_elm.append((self.Jx + self.Jy)*self.i);
                else:
                    if double_hopping:
                        lind_states.append([ [pos, right_neighbour], None ]);
                        lind_elm.append((self.Jx - self.Jy)*self.i);
                    
                # Off-diagonal term from B (hopping and double hopping) -> ROW
                if S_i[pos] != S_i[down_neighbour]:
                    lind_states.append([ [pos, down_neighbour], None ]);
                    lind_elm.append((self.Jx + self.Jy)*self.i);
                else:
                    if double_hopping:
                        lind_states.append([ [pos, down_neighbour], None ]);
                        lind_elm.append((self.Jx - self.Jy)*self.i);
                           
                # Off-diagonal term from external field Sx
                if external:
                    lind_states.append([[ pos ],None]);
                    lind_elm.append( self.i *self.field );
   
                      
                ######################################################################
                ######################################################################
                    
                #Term C -Does not have diagonal term. Flips pos in S and S' iff [pos] == -1 in both
                if (S_i[pos] != 1) and (S_j[pos] != 1):
                    lind_elm.append(self.gamma);
                    lind_states.append([ [pos] , [pos] ]);
                     
                ######################################################################
                ######################################################################
                    
                # Term D - Only has diagonal term. Iff S_2[pos] == 1
                if S_j[pos] == 1:
                    lind_elm[0] += -0.5*self.gamma;
                         
                ######################################################################
                ######################################################################
                     
                #Term E - Only has diagonal term. Iff S_1[pos] == 1
                if S_i[pos] == 1:
                    lind_elm[0] += -0.5*self.gamma;
        
        
        return lind_elm, lind_states;
    
    
    # dictionary for Lindbladian move generation
    def InitMoveDict(self):     
        
    #########################################################
    # Defining all the Linbladian move functions for the dictionary
    # Python does not have switch-case statements    
    #########################################################
        def hopping_column_i(pos, S_i, S_j):
            return [pos, self.CalcRightNN(pos,self.X_site,self.Y_site)], None, 1;
    
        def hopping_row_i(pos, S_i, S_j):
            return [pos, self.CalcDownNN(pos,self.X_site,self.Y_site)], None, 1;
    
        def hopping_column_j(pos, S_i, S_j):
            return None, [pos, self.CalcRightNN(pos,self.X_site,self.Y_site)], 1;
    
        def hopping_row_j(pos, S_i, S_j):
            return None, [pos, self.CalcDownNN(pos,self.X_site,self.Y_site)], 1;
        
        def external_i(pos, S_i, S_j):
            return pos, None, 1;
        
        def external_j(pos, S_i, S_j):
            return None, pos, 1;
    
        def dissipator(pos, S_i, S_j):
            if S_i[pos] == S_j[pos]: 
                if S_i[pos] == 1:
                    return pos, pos, 0.1;
                else:
                    if random.random() < 0.1:
                        return pos, pos, 10;
                    else:
                        return None, None, 1;                                
            else:
                return None, None, 1;
            
        def jumper(pos, S_i, S_j):
            pos_i = random.randrange(0, self.N_site);
            pos_j = random.randrange(0, self.N_site);
            
            return pos_i, pos_j, 1;
                        
             
    #########################################################         
        self.moveGen = {
            0: hopping_column_i,
            1: hopping_row_i,
            2: hopping_column_j,
            3: hopping_row_j,
            4: dissipator,
            5: external_i,
            6: external_j,
            7: jumper
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

        
    
    
    
