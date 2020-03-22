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
 # Observable_Sampler.py
 #
 #  Created on: May 23, 2018
 #      Author: Alexandra Nagy
 #              alex.nagy.phys@gmail.com
 #
 #      REFERENCES:
 #      - A. Nagy, V. Savona, Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems, Physical review letters 122 (25), 250501, 2019
 #      - A. Nagy's PhD Dissertation, École polytechnique fédérale de Lausanne, Quantum Monte Carlo Approach to the non-equilibrium steady state of open quantum systems, 2020
 #    
from Classes.NQS import NQS
from Classes.Heisenberg_2D_XYZ import Heisenberg_2D_XYZ
from Classes.Heisenberg_2D_XYZ_Hz import Heisenberg_2D_XYZ_Hz
import numpy as np
import sys
import os
from Classes.Metropolis_Hastings import Metropolis_Hastings
from Classes.Observable import Observable

def main():
    
 #######################################################################
    # Parameters coming from command line
    N_x = int(sys.argv[1]);
    N_y = int(sys.argv[2]);
    hidden_ratio = int(sys.argv[3]);
    hidden_mixing = int(sys.argv[4]);
    Jx = float(sys.argv[5]);
    Jy = float(sys.argv[6]);
    Jz = float(sys.argv[7]);
    gamma = float(sys.argv[8]);
    field = float(sys.argv[9]);
    MC_sweep = int(sys.argv[10]);
    thermal_rate = float(sys.argv[11]);
    input = sys.argv[12];
    output = sys.argv[13];
    ini = int(sys.argv[14]);
    fin = int(sys.argv[15]);
#######################################################################

    # Initialize NQS
    nqs = NQS(node_visible = N_x*N_y, node_hidden = hidden_ratio*N_x*N_y, node_mixing = hidden_mixing*N_x*N_y);
    # Initialize Model
    lind = Heisenberg_2D_XYZ_Hz(N_x, N_y, Jx, Jy, Jz, gamma, field);
    lind.InitMoveDict();   
    # Initialize Observable
    obs = Observable(N_x*N_y);
    # Initialize MH
    mh = Metropolis_Hastings(nqs, lind, thermal_rate, MC_sweep, obs);

    # read in all the lines
    file = open(input,'r');
    data = file.readlines();
    file.close();
  
    for line in range(len(data)):
        if line >= ini and line < fin:

            print("..........    {}   ..........".format(line-ini));
            nqs.param_set(data[line]);
            mh.initState();
            obs.reset_local();  
            
            mh.thermalization(lind.field == 0);
            mh.sampling(lind.field == 0);
            
            
    obs.DivideGlobal(fin-ini);
    print("Sigma Z:  {}   +/-   {}".format((1/obs.nsite)*np.squeeze(np.sum(obs.Sigma_z)), (1/obs.nsite)*np.sqrt(np.sum(obs.Var_Sigma_z**2))));
    print("Sigma X:  {}   +/-   {}".format((1/obs.nsite)*np.squeeze(np.sum(obs.Sigma_x)), (1/obs.nsite)*np.sqrt(np.sum(obs.Var_Sigma_x**2))));
    print("Sigma Y:  {}   +/-   {}".format((1/obs.nsite)*np.squeeze(np.sum(obs.Sigma_y)), (1/obs.nsite)*np.sqrt(np.sum(obs.Var_Sigma_y**2))));
    
    # write out result
    file = open(output, 'wb');
    #file.write("Sigma Z , Sigma X, Sigma Y, sigma ZZ, sigma XX, sigma YY, then errors beneath each other\n");
    np.savetxt(file, obs.Sigma_z.T, fmt='%1.4e');
    np.savetxt(file, obs.Sigma_x.T, fmt='%1.4e');
    np.savetxt(file, obs.Sigma_y.T, fmt='%1.4e');
    np.savetxt(file, obs.Sigma_ZZ, fmt='%1.4e');
    np.savetxt(file, obs.Sigma_XX, fmt='%1.4e');
    np.savetxt(file, obs.Sigma_YY, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_z.T, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_x.T, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_y.T, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_ZZ, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_XX, fmt='%1.4e');
    np.savetxt(file, obs.Var_Sigma_YY, fmt='%1.4e');
    file.close();



main();
