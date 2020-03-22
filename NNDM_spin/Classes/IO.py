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
 # IO.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #

# for outputting the results
# in the name:
#         - D : data of the simulation
#         - E : estimators at each iteration
#         - P : parameters of the network at each iteration

# Outputting data of the model and the network

# NOTICE!!!!!!!!!!!!!!!!!
# savetxt is patched. there was a bug writing out (6-7i) as (6+-7i). 
# the patch:  https://github.com/numpy/numpy/pull/10875.patch
# BUT
# this is NOT compatible with python3
# gotta modify original library by hand!!
# for every compilation ona  new machine!

import os
from numpy import savetxt, real, imag, squeeze

class IO(object):
    
    # not in __init__ because of module import reasons
    def Initialize(self, nqs, lindbladian, sweep_MC, thermal_rate, learning_rate, path, method, lru, drawn):
        #generate output directory
        self.name =   lindbladian.model_name + '_' + str(lindbladian.N_site) \
                        + '_Jx' + str(lindbladian.Jx) \
                        + '_Jy' + str(lindbladian.Jy) \
                        + '_Jz' + str(lindbladian.Jz) \
                        + '_h' + str(lindbladian.field) \
                        + '_H' + str(nqs.node_hidden) \
                        + '_M' + str(nqs.node_mixing);
        self.data_dir = path + self.name;
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir);   
            
        # output simulation parameters
        data = open(self.data_dir + '/D_' + self.name + '.txt', 'w');
        data.write('#################################\n');
        data.write('Method\n');
        data.write('#################################\n');
        data.write('Method:  {}\n'.format(method));
        if method == 'ldag_y':
            data.write('Drawn number:  {}\n'.format(drawn));
        else:
            data.write('LRU size:  {}\n'.format(lru));
        data.write('#################################\n');
        data.write('Model\n');
        data.write('#################################\n');
        data.write('Model:  {}\n'.format(lindbladian.model_name));
        data.write('N_site:  {}\n'.format(lindbladian.N_site));
        data.write('X_site:  {}\n'.format(lindbladian.X_site));
        data.write('Y_site:  {}\n'.format(lindbladian.Y_site));
        data.write('J_x:  {}\n'.format(lindbladian.Jx));
        data.write('J_y:  {}\n'.format(lindbladian.Jy));
        data.write('J_z:  {}\n'.format(lindbladian.Jz));
        data.write('Gamma:  {}\n'.format(lindbladian.gamma));
        data.write('Field x:  {}\n'.format(lindbladian.field));
        data.write('#################################\n');
        data.write('#################################\n');
        data.write('Network\n');
        data.write('#################################\n');
        data.write('Node visible:  {}\n'.format(nqs.node_visible));
        data.write('Node hidden:  {}\n'.format(nqs.node_hidden));
        data.write('Node mixing:  {}\n'.format(nqs.node_mixing));
        data.write('Parameter number:  {}\n'.format(nqs.paramNbr));
        data.write('Thermal rate:  {}\n'.format(thermal_rate));
        data.write('Met-Hast sweeps:  {}\n'.format(sweep_MC));
        data.write('Learning rate:  {}\n'.format(learning_rate));
        
        data.close();
        
        # cleaning files for parameters and estimators
        self.P_name = self.data_dir + '/P_' + self.name + '.txt';
        param = open(self.P_name, 'w');
        param.write('');
        param.close();
        self.E_name = self.data_dir + '/E_' + self.name + '.txt'; 
        estim = open(self.E_name, 'w');
        estim.write('');
        estim.close();
        self.Prof_name = self.data_dir + '/Profile.txt';
        prof = open(self.Prof_name, 'w');
        prof.write('');
        prof.close();
        
    # Output parameter values
    def Output_Param(self, nqs):

        P = open(self.P_name, 'ab');
        savetxt( P, squeeze(real(nqs.a.T)), newline = ' ', delimiter = '', header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(imag(nqs.a.T)), newline = ' ', delimiter = '', header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(real(nqs.b.T)), newline = ' ', delimiter = '',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(imag(nqs.b.T)), newline = ' ', delimiter = '',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(real(nqs.X.reshape(nqs.node_hidden*nqs.node_visible,1))), newline = ' ',delimiter = '',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(imag(nqs.X.reshape(nqs.node_hidden*nqs.node_visible,1))), newline = ' ',delimiter = '',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(real(nqs.bp.T)), newline = ' ', delimiter = ' ',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(real(nqs.W.reshape(nqs.node_mixing*nqs.node_visible,1))), newline = ' ',delimiter = '',header ='', footer ='', fmt = '%1.4e');
        savetxt( P, squeeze(imag(nqs.W.reshape(nqs.node_mixing*nqs.node_visible,1))), newline = ' ',delimiter = '',header ='', footer ='', fmt = '%1.4e');
        P.write(str.encode('\n'));
        P.close();
        
    
    # Output estimators
    def Output_Estimator(self, iter, Lrho, Lrho_var):
        
        E = open(self.E_name, 'a');
        E.write('{0}\t{1}\t{2}\n'.format(iter,real(Lrho),Lrho_var));
        E.close();
    
    def Output_Profiling(self, move_rate, lru_usage, MH, comm, SR_solver):
        
        Prof = open(self.Prof_name, 'a');
        Prof.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(move_rate,lru_usage, MH, comm, SR_solver));
        Prof.close();
        
    
    
    
    
