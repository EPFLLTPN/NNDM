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
 # Observable.py
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

class Observable(object):
    
    def __init__(self, nsite):
        self.nsite = nsite;
        self.Sigma_z = np.zeros((self.nsite,1));
        self.Var_Sigma_z = np.zeros((self.nsite,1));
        self.Sigma_x = np.zeros((self.nsite,1));
        self.Var_Sigma_x = np.zeros((self.nsite,1));
        self.Sigma_y = np.zeros((self.nsite,1));
        self.Var_Sigma_y = np.zeros((self.nsite,1));
        
        self.Sigma_ZZ = np.zeros((self.nsite,self.nsite));
        self.Var_Sigma_ZZ = np.zeros((self.nsite,self.nsite));
        self.Sigma_XX = np.zeros((self.nsite,self.nsite));
        self.Var_Sigma_XX = np.zeros((self.nsite,self.nsite));
        self.Sigma_YY = np.zeros((self.nsite,self.nsite));
        self.Var_Sigma_YY = np.zeros((self.nsite,self.nsite));
        
    def reset_local(self):
        self.rho = 0.0;
        self.var_rho = 0.0;
        self.sz = np.zeros((self.nsite,1),dtype=np.complex_);
        self.var_sz = np.zeros((self.nsite,1),dtype=np.complex_);
        self.sx = np.zeros((self.nsite,1),dtype=np.complex_);
        self.var_sx = np.zeros((self.nsite,1),dtype=np.complex_);
        self.sy = np.zeros((self.nsite,1),dtype=np.complex_);
        self.var_sy = np.zeros((self.nsite,1),dtype=np.complex_);
        
        self.szz = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
        self.var_szz = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
        self.sxx = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
        self.var_sxx = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
        self.syy = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
        self.var_syy = np.zeros((self.nsite,self.nsite),dtype=np.complex_);
         
    def calculate_local(self, S_i, S_j, iter, rho):
    # with running average and variance    
        delta = self.Div_Rho(S_i, S_j, rho) - self.rho;
        self.rho += delta/(iter+1);
        delta2 = self.Div_Rho(S_i, S_j, rho) - self.rho;
        self.var_rho += delta*delta2;    
        
    # Sigma_z
        delta = self.Sigma_Z_value(S_i, S_j, self.nsite, rho) - self.sz;
        self.sz += delta/(iter+1);
        delta2 = self.Sigma_Z_value(S_i, S_j, self.nsite, rho) - self.sz;
        self.var_sz += np.multiply(delta,delta2);

    # Sigma_x
        delta = self.Sigma_X_value(S_i, S_j, self.nsite, rho) - self.sx;
        self.sx += delta/(iter+1);
        delta2 = self.Sigma_X_value(S_i, S_j, self.nsite, rho) - self.sx;
        self.var_sx += np.multiply(delta,delta2);

    # Sigma_y
        delta = self.Sigma_Y_value(S_i, S_j, self.nsite, rho) - self.sy;
        self.sy += delta/(iter+1);
        delta2 = self.Sigma_Y_value(S_i, S_j, self.nsite, rho) - self.sy;
        self.var_sy += np.multiply(delta,delta2);
        
    # Sigma_zz
        delta = self.Sigma_ZZ_value(S_i, S_j, self.nsite, rho) - self.szz;
        self.szz += delta/(iter+1);
        delta2 = self.Sigma_ZZ_value(S_i, S_j, self.nsite, rho) - self.szz;
        self.var_szz += np.multiply(delta,delta2);
        
    # Sigma_xx
        delta = self.Sigma_XX_value(S_i, S_j, self.nsite, rho) - self.sxx;
        self.sxx += delta/(iter+1);
        delta2 = self.Sigma_XX_value(S_i, S_j, self.nsite, rho) - self.sxx;
        self.var_sxx += np.multiply(delta,delta2);

    # Sigma_yy
        delta = self.Sigma_YY_value(S_i, S_j, self.nsite, rho) - self.syy;
        self.syy += delta/(iter+1);
        delta2 = self.Sigma_YY_value(S_i, S_j, self.nsite, rho) - self.syy;
        self.var_syy += np.multiply(delta,delta2);

        
            
    def Divide(self, sweeps):
        self.var_sz = np.sqrt(self.var_sz)/(sweeps - 1);
        self.var_sx = np.sqrt(self.var_sx)/(sweeps - 1);
        self.var_sy = np.sqrt(self.var_sy)/(sweeps - 1);
        self.var_szz = np.sqrt(self.var_szz)/(sweeps - 1);
        self.var_sxx = np.sqrt(self.var_sxx)/(sweeps - 1);
        self.var_syy = np.sqrt(self.var_syy)/(sweeps - 1);
        self.var_rho = np.sqrt(self.var_rho)/(sweeps - 1);
        
        self.var_sz = np.real(self.sz/self.rho * np.sqrt( np.square(self.var_sz/self.sz) + np.square(self.var_rho/self.rho)));
        self.sz = np.real(self.sz/self.rho);
        
        self.var_sx = np.real(self.sx/self.rho * np.sqrt( np.square(self.var_sx/self.sx) + np.square(self.var_rho/self.rho)));
        self.sx = np.real(self.sx/self.rho);
        
        self.var_sy = np.real(self.sy/self.rho * np.sqrt( np.square(self.var_sy/self.sy) + np.square(self.var_rho/self.rho)));
        self.sy = np.real(self.sy/self.rho);
              
        self.var_szz = np.real(self.szz/self.rho * np.sqrt( np.square(self.var_szz/self.szz) + np.square(self.var_rho/self.rho)));
        self.szz = np.real(self.szz/self.rho);
        
        self.var_sxx = np.real(self.sxx/self.rho * np.sqrt( np.square(self.var_sxx/self.sxx) + np.square(self.var_rho/self.rho)));
        self.sxx = np.real(self.sxx/self.rho);
        
        self.var_syy = np.real(self.syy/self.rho * np.sqrt( np.square(self.var_syy/self.syy) + np.square(self.var_rho/self.rho)));
        self.syy = np.real(self.syy/self.rho);

    
    def Accumulate_global(self):
        self.Sigma_z += self.sz;
        self.Var_Sigma_z += np.square(self.var_sz);
        self.Sigma_x += self.sx;
        self.Var_Sigma_x += np.square(self.var_sx);
        self.Sigma_y += self.sy;
        self.Var_Sigma_y += np.square(self.var_sy);
        self.Sigma_ZZ += self.szz;
        self.Var_Sigma_ZZ += np.square(self.var_szz);
        self.Sigma_XX += self.sxx;
        self.Var_Sigma_XX += np.square(self.var_sxx);
        self.Sigma_YY += self.syy;
        self.Var_Sigma_YY += np.square(self.var_syy);
        
        
    def DivideGlobal(self, lines):
        self.Sigma_z /= lines;
        self.Var_Sigma_z = np.sqrt(self.Var_Sigma_z)/lines;
        self.Sigma_x /= lines;
        self.Var_Sigma_x = np.sqrt(self.Var_Sigma_x)/lines;
        self.Sigma_y /= lines;
        self.Var_Sigma_y = np.sqrt(self.Var_Sigma_y)/lines;
        
        self.Sigma_ZZ /= lines;
        self.Var_Sigma_ZZ = np.sqrt(self.Var_Sigma_ZZ)/lines;
        self.Sigma_XX /= lines;
        self.Var_Sigma_XX = np.sqrt(self.Var_Sigma_XX)/lines;
        self.Sigma_YY /= lines;
        self.Var_Sigma_YY = np.sqrt(self.Var_Sigma_YY)/lines;
        
    
    @staticmethod
    def Sigma_Z_value(S_i, S_j, nsite, rho):
        if (S_i == S_j).all():
            return S_i.copy()*(1/rho);
        else:
            return np.zeros((nsite,1),dtype=np.complex_);
        
    @staticmethod
    def Div_Rho(S_i, S_j, rho):
        if (S_i == S_j).all():
            return 1/rho;
        else:
            return 0.0;
        
    @staticmethod
    def Sigma_X_value(S_i, S_j, nsite, rho):
        if np.count_nonzero(S_i^S_j) == 1:
            sx = np.zeros((nsite,1),dtype=np.complex_);
            where = np.nonzero(S_i^S_j)[0];
            sx[where] = (1/rho);
            return sx;
        else:
            return np.zeros((nsite,1),dtype=np.complex_);
        
    @staticmethod
    def Sigma_Y_value(S_i, S_j, nsite, rho):
        if np.count_nonzero(S_i^S_j) == 1:
            sy = np.zeros((nsite,1),dtype=np.complex_);
            where = np.nonzero(S_i^S_j)[0];
            sy[where] = S_i[where]*1j*(1/rho);
            return sy;
        else:
            return np.zeros((nsite,1),dtype=np.complex_);
        
        
    @staticmethod
    def Sigma_ZZ_value(S_i, S_j, nsite, rho):
        if (S_i == S_j).all():
            return np.dot(S_i, S_j.T)*(1/rho);
        else:
            return np.zeros((nsite,nsite),dtype=np.complex_);
        
    @staticmethod
    def Sigma_XX_value(S_i, S_j, nsite, rho):
        if np.count_nonzero(S_i^S_j) == 2:
            sx = np.zeros((nsite,nsite),dtype=np.complex_);
            where = np.nonzero(S_i^S_j)[0];
            sx[where[0],where[1]] = (1/rho);
            sx[where[1],where[0]] = (1/rho);
            return sx;
        else:
            return np.zeros((nsite,nsite),dtype=np.complex_);
        
    @staticmethod
    def Sigma_YY_value(S_i, S_j, nsite, rho):
        if np.count_nonzero(S_i^S_j) == 2:
            sy = np.zeros((nsite,nsite),dtype=np.complex_);
            where = np.nonzero(S_i^S_j)[0];
            sy[where[0],where[1]] = np.squeeze((-1)*S_i[where[0]]*S_j[where[1]]*(1/rho));
            sy[where[1],where[0]] = np.squeeze((-1)*S_i[where[0]]*S_j[where[1]]*(1/rho));
            return sy;
        else:
            return np.zeros((nsite,nsite),dtype=np.complex_);
        
        
        
        
    
