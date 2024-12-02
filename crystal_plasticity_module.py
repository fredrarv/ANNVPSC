import sys
import numpy as np
import itertools
import scipy.optimize
import scipy.linalg
import time
#import quadprog
import matplotlib.pyplot as plt
import ipywidgets
from multiprocessing import Pool
from IPython.display import display
import odflib
np.set_printoptions(threshold=np.inf) 

global ImportErrCpfort

try:
    import cpfort
    ImportErrCpfort = False
except:
    print('Cannot import cpfort module for fortran accelerated Alamel calculations. Python Alamel will be run.')
    ImportErrCpfort = True

outputvars_dict = {'average_stress':1,
                   'average_slip'  :2,
                   'stress_loc'    :3,
                   'stress_glob'   :4,
                   'euler_angles'  :5,
                   'crss'          :6,
                   'sliprates'     :7,
                   'activesID'     :8,
                   'num_actives'   :9,
                   'statevar'      :10,
                   'relaxation'    :11 }

grain_interaction_dict = {'FCTAYLOR':1,
                          'ALAMEL'  :2,
                          'ALAMEL3' :3}

hardening_model_dict = {'RIGID_PLASTIC':1,
                        'BAUSCHINGER'  :2}

crystal_structure_dict = {'FCC_111':1,
                          'FCC_110':12,
                          'FCC_100':123,
                          'FCC_112':1234,
                          'BCC_110':5,
                          'BCC_112':6,
                          'BCC_123':7}


# definition of the Cluster class
class Cluster:
    def __init__(self, grain1, grain2, grain_boundary):
        self.g1 = grain1
        self.g2 = grain2
        self.gb = grain_boundary
        
        
    def findslips_relaxed_old(self, Pg1, Pg2, Dp):
    # ineffective formulation in grain boundary system, requiring rotation of Schmid matrices P into this system for both grains
    # use "findslips_relaxed" instead
            
        f = np.zeros(52)
        f[:24]   = self.g1.crss
        f[24:48] = self.g2.crss
        f[48:50] = self.g1.relax_penalties[:2]
        f[50:52] = self.g2.relax_penalties[:2]
        
                   
        A_eq = np.array([[Pg1[0,0,0],  Pg1[0,0,1],  Pg1[0,0,2],  Pg1[0,0,3],   Pg1[0,0,4],   Pg1[0,0,5], 
                          Pg1[0,0,6],  Pg1[0,0,7],  Pg1[0,0,8],  Pg1[0,0,9],   Pg1[0,0,10],  Pg1[0,0,11],
                          Pg1[0,0,12], Pg1[0,0,13], Pg1[0,0,14], Pg1[0,0,15],  Pg1[0,0,16],  Pg1[0,0,17], 
                          Pg1[0,0,18], Pg1[0,0,19], Pg1[0,0,20], Pg1[0,0,21],  Pg1[0,0,22],  Pg1[0,0,23], 
                          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          crystal_properties.symR1[0,0], 
                          crystal_properties.symR2[0,0], 
                         -crystal_properties.symR1[0,0], 
                         -crystal_properties.symR2[0,0]],

                         [Pg1[1,1,0],  Pg1[1,1,1],  Pg1[1,1,2],  Pg1[1,1,3],   Pg1[1,1,4],   Pg1[1,1,5], 
                          Pg1[1,1,6],  Pg1[1,1,7],  Pg1[1,1,8],  Pg1[1,1,9],   Pg1[1,1,10],  Pg1[1,1,11],
                          Pg1[1,1,12], Pg1[1,1,13], Pg1[1,1,14], Pg1[1,1,15],  Pg1[1,1,16],  Pg1[1,1,17], 
                          Pg1[1,1,18], Pg1[1,1,19], Pg1[1,1,20], Pg1[1,1,21],  Pg1[1,1,22],  Pg1[1,1,23],  
                          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  
                          crystal_properties.symR1[1,1],   
                          crystal_properties.symR2[1,1],  
                         -crystal_properties.symR1[1,1],  
                         -crystal_properties.symR2[1,1]],

                         [Pg1[1,2,0],  Pg1[1,2,1],  Pg1[1,2,2],  Pg1[1,2,3],   Pg1[1,2,4],   Pg1[1,2,5], 
                          Pg1[1,2,6],  Pg1[1,2,7],  Pg1[1,2,8],  Pg1[1,2,9],   Pg1[1,2,10],  Pg1[1,2,11],
                          Pg1[1,2,12], Pg1[1,2,13], Pg1[1,2,14], Pg1[1,2,15],  Pg1[1,2,16],  Pg1[1,2,17], 
                          Pg1[1,2,18], Pg1[1,2,19], Pg1[1,2,20], Pg1[1,2,21],  Pg1[1,2,22],  Pg1[1,2,23], 
                          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  
                          crystal_properties.symR1[1,2],   
                          crystal_properties.symR2[1,2],  
                         -crystal_properties.symR1[1,2],  
                         -crystal_properties.symR2[1,2]],
          
                         [Pg1[0,2,0],  Pg1[0,2,1],  Pg1[0,2,2],  Pg1[0,2,3],   Pg1[0,2,4],   Pg1[0,2,5], 
                          Pg1[0,2,6],  Pg1[0,2,7],  Pg1[0,2,8],  Pg1[0,2,9],   Pg1[0,2,10],  Pg1[0,2,11],
                          Pg1[0,2,12], Pg1[0,2,13], Pg1[0,2,14], Pg1[0,2,15],  Pg1[0,2,16],  Pg1[0,2,17], 
                          Pg1[0,2,18], Pg1[0,2,19], Pg1[0,2,20], Pg1[0,2,21],  Pg1[0,2,22],  Pg1[0,2,23], 
                          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  
                          crystal_properties.symR1[0,2],   
                          crystal_properties.symR2[0,2],  
                         -crystal_properties.symR1[0,2],  
                         -crystal_properties.symR2[0,2]],
          
                         [Pg1[0,1,0],  Pg1[0,1,1],  Pg1[0,1,2],  Pg1[0,1,3],   Pg1[0,1,4],   Pg1[0,1,5], 
                          Pg1[0,1,6],  Pg1[0,1,7],  Pg1[0,1,8],  Pg1[0,1,9],   Pg1[0,1,10],  Pg1[0,1,11],
                          Pg1[0,1,12], Pg1[0,1,13], Pg1[0,1,14], Pg1[0,1,15],  Pg1[0,1,16],  Pg1[0,1,17], 
                          Pg1[0,1,18], Pg1[0,1,19], Pg1[0,1,20], Pg1[0,1,21],  Pg1[0,1,22],  Pg1[0,1,23], 
                          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  
                          crystal_properties.symR1[0,1],   
                          crystal_properties.symR2[0,1],  
                         -crystal_properties.symR1[0,1],  
                         -crystal_properties.symR2[0,1]],
          
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          Pg2[0,0,0],  Pg2[0,0,1],  Pg2[0,0,2],  Pg2[0,0,3],   Pg2[0,0,4],   Pg2[0,0,5], 
                          Pg2[0,0,6],  Pg2[0,0,7],  Pg2[0,0,8],  Pg2[0,0,9],   Pg2[0,0,10],  Pg2[0,0,11],
                          Pg2[0,0,12], Pg2[0,0,13], Pg2[0,0,14], Pg2[0,0,15],  Pg2[0,0,16],  Pg2[0,0,17], 
                          Pg2[0,0,18], Pg2[0,0,19], Pg2[0,0,20], Pg2[0,0,21],  Pg2[0,0,22],  Pg2[0,0,23], 
                         -crystal_properties.symR1[0,0], 
                         -crystal_properties.symR2[0,0], 
                          crystal_properties.symR1[0,0], 
                          crystal_properties.symR2[0,0]],

                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          Pg2[1,1,0],  Pg2[1,1,1],  Pg2[1,1,2],  Pg2[1,1,3],   Pg2[1,1,4],   Pg2[1,1,5], 
                          Pg2[1,1,6],  Pg2[1,1,7],  Pg2[1,1,8],  Pg2[1,1,9],   Pg2[1,1,10],  Pg2[1,1,11],
                          Pg2[1,1,12], Pg2[1,1,13], Pg2[1,1,14], Pg2[1,1,15],  Pg2[1,1,16],  Pg2[1,1,17], 
                          Pg2[1,1,18], Pg2[1,1,19], Pg2[1,1,20], Pg2[1,1,21],  Pg2[1,1,22],  Pg2[1,1,23],    
                         -crystal_properties.symR1[1,1],   
                         -crystal_properties.symR2[1,1],  
                          crystal_properties.symR1[1,1],  
                          crystal_properties.symR2[1,1]],

                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          Pg2[1,2,0],  Pg2[1,2,1],  Pg2[1,2,2],  Pg2[1,2,3],   Pg2[1,2,4],   Pg2[1,2,5], 
                          Pg2[1,2,6],  Pg2[1,2,7],  Pg2[1,2,8],  Pg2[1,2,9],   Pg2[1,2,10],  Pg2[1,2,11],
                          Pg2[1,2,12], Pg2[1,2,13], Pg2[1,2,14], Pg2[1,2,15],  Pg2[1,2,16],  Pg2[1,2,17], 
                          Pg2[1,2,18], Pg2[1,2,19], Pg2[1,2,20], Pg2[1,2,21],  Pg2[1,2,22],  Pg2[1,2,23],     
                         -crystal_properties.symR1[1,2],   
                         -crystal_properties.symR2[1,2],  
                          crystal_properties.symR1[1,2],  
                          crystal_properties.symR2[1,2]],
          
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          Pg2[0,2,0],  Pg2[0,2,1],  Pg2[0,2,2],  Pg2[0,2,3],   Pg2[0,2,4],   Pg2[0,2,5], 
                          Pg2[0,2,6],  Pg2[0,2,7],  Pg2[0,2,8],  Pg2[0,2,9],   Pg2[0,2,10],  Pg2[0,2,11],
                          Pg2[0,2,12], Pg2[0,2,13], Pg2[0,2,14], Pg2[0,2,15],  Pg2[0,2,16],  Pg2[0,2,17], 
                          Pg2[0,2,18], Pg2[0,2,19], Pg2[0,2,20], Pg2[0,2,21],  Pg2[0,2,22],  Pg2[0,2,23],       
                         -crystal_properties.symR1[0,2],   
                         -crystal_properties.symR2[0,2],  
                          crystal_properties.symR1[0,2],  
                          crystal_properties.symR2[0,2]],
          
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                          Pg2[0,1,0],  Pg2[0,1,1],  Pg2[0,1,2],  Pg2[0,1,3],   Pg2[0,1,4],   Pg2[0,1,5], 
                          Pg2[0,1,6],  Pg2[0,1,7],  Pg2[0,1,8],  Pg2[0,1,9],   Pg2[0,1,10],  Pg2[0,1,11],
                          Pg2[0,1,12], Pg2[0,1,13], Pg2[0,1,14], Pg2[0,1,15],  Pg2[0,1,16],  Pg2[0,1,17], 
                          Pg2[0,1,18], Pg2[0,1,19], Pg2[0,1,20], Pg2[0,1,21],  Pg2[0,1,22],  Pg2[0,1,23], 
                         -crystal_properties.symR1[0,1],   
                         -crystal_properties.symR2[0,1],  
                          crystal_properties.symR1[0,1],  
                          crystal_properties.symR2[0,1]]])
          
        Dp_vec = m2voigt_dev(Dp)
        b_eq = np.append(Dp_vec, Dp_vec)

        sol = scipy.optimize.linprog(f, A_ub=None, b_ub=None, A_eq=A_eq, 
                                     b_eq=b_eq, bounds=None, method='revised simplex', callback=None)   # bounds=None means by default positive solution only
        if sol.status != 0: # simplex failed to find a solution
            print(sol.message)
            print('Trying interior-point algorithm...')
            sol = scipy.optimize.linprog(f, A_ub=None, b_ub=None, A_eq=A_eq, 
                                     b_eq=b_eq, bounds=None, method='interior-point', callback=None)
            print(sol.message)
            
        
        inds = np.flip(np.argsort(sol.x[:-4]))[:8]
        self.g1.activesID = inds[inds < 24]
        self.g2.activesID = inds[inds >= 24] - 24
        
        self.g1.sliprates = np.zeros(24)
        self.g2.sliprates = np.zeros(24)
        
        self.g1.sliprates[self.g1.activesID] = sol.x[self.g1.activesID]
        self.g2.sliprates[self.g2.activesID] = sol.x[self.g2.activesID+24]
            
        self.g1.relaxation_sliprates[0] = sol.x[48]-sol.x[50] # L_gm23
        self.g1.relaxation_sliprates[1] = sol.x[49]-sol.x[51] # L_gm13
        self.g2.relaxation_sliprates = -self.g1.relaxation_sliprates
      
    def findslips_relaxed(self, symR1g1, symR2g1, symR1g2, symR2g2):
         
        f = np.zeros(52)
        f[:24]   = self.g1.crss
        f[24:48] = self.g2.crss
        f[48:50] = self.g1.relax_penalties[:2]
        f[50:52] = self.g2.relax_penalties[:2]
                   
        A_eq = np.zeros((10,52))
        A_eq[:5,:24]   = crystal_properties.P
        A_eq[5:,24:48] = crystal_properties.P

        A_eq[:5,48] = symR1g1
        A_eq[:5,49] = symR2g1
        A_eq[:5,50] = -symR1g1
        A_eq[:5,51] = -symR2g1
        
        A_eq[5:,48] = -symR1g2
        A_eq[5:,49] = -symR2g2
        A_eq[5:,50] = symR1g2
        A_eq[5:,51] = symR2g2
          
        b_eq = np.append(self.g1.Dp_vec, self.g2.Dp_vec)

        sol = scipy.optimize.linprog(f, A_ub=None, b_ub=None, A_eq=A_eq, 
                                     b_eq=b_eq, bounds=None, method='revised simplex', callback=None)   # bounds=None means by default positive solution only
        if sol.status != 0: # simplex failed to find a solution
            print(sol.message)
            print('Trying interior-point algorithm...')
            sol = scipy.optimize.linprog(f, A_ub=None, b_ub=None, A_eq=A_eq, 
                                     b_eq=b_eq, bounds=None, method='interior-point', callback=None)
            print(sol.message)
            
        
        inds = np.flip(np.argsort(sol.x[:-4]))[:8]
        self.g1.activesID = inds[inds < 24]
        self.g2.activesID = inds[inds >= 24] - 24
        
        self.g1.sliprates = np.zeros(24)
        self.g2.sliprates = np.zeros(24)
        
        self.g1.sliprates[self.g1.activesID] = sol.x[self.g1.activesID]
        self.g2.sliprates[self.g2.activesID] = sol.x[self.g2.activesID+24]
            
        self.g1.relaxation_sliprates[0] = sol.x[48]-sol.x[50] # L_gm13
        self.g1.relaxation_sliprates[1] = sol.x[49]-sol.x[51] # L_gm23
        self.g2.relaxation_sliprates = -self.g1.relaxation_sliprates

            
    def getstress(self, symR1g1, symR2g1, symR1g2, symR2g2):
            
        B = np.zeros((10,10))
        crss_cluster = np.zeros(10)
        # grain1
        N1 = len(self.g1.activesID)
        for i, ind in enumerate(self.g1.activesID):
            B[i,:5] = np.array([2.*crystal_properties.P[0,ind] + crystal_properties.P[1,ind], 
                                2.*crystal_properties.P[1,ind] + crystal_properties.P[0,ind],
                                2.*crystal_properties.P[2,ind], 
                                2.*crystal_properties.P[3,ind],
                                2.*crystal_properties.P[4,ind]])       
        # grain2
        N2 = len(self.g2.activesID)
        for i, ind in enumerate(self.g2.activesID):
            B[N1+i,5:] = np.array([2.*crystal_properties.P[0,ind] + crystal_properties.P[1,ind], 
                                   2.*crystal_properties.P[1,ind] + crystal_properties.P[0,ind],
                                   2.*crystal_properties.P[2,ind], 
                                   2.*crystal_properties.P[3,ind],
                                   2.*crystal_properties.P[4,ind]])   
        # relaxations      
        B[N1+N2,:]   = [ 2.*symR1g1[0] + symR1g1[1], 
                         2.*symR1g1[1] + symR1g1[0], 
                         2.*symR1g1[2], 
                         2.*symR1g1[3], 
                         2.*symR1g1[4],
                        -2.*symR1g2[0] - symR1g2[1], 
                        -2.*symR1g2[1] - symR1g2[0], 
                        -2.*symR1g2[2], 
                        -2.*symR1g2[3], 
                        -2.*symR1g2[4] ]
        
        B[N1+N2+1,:] = [ 2.*symR2g1[0] + symR2g1[1], 
                         2.*symR2g1[1] + symR2g1[0], 
                         2.*symR2g1[2], 
                         2.*symR2g1[3], 
                         2.*symR2g1[4],
                        -2.*symR2g2[0] - symR2g2[1], 
                        -2.*symR2g2[1] - symR2g2[0], 
                        -2.*symR2g2[2], 
                        -2.*symR2g2[3], 
                        -2.*symR2g2[4] ]     

        crss_cluster[:N1]      = self.g1.crss[self.g1.activesID]
        crss_cluster[N1:N1+N2] = self.g2.crss[self.g2.activesID]
        x = np.linalg.solve(B, crss_cluster)

        # express stress into the each grain's coord sys and global sys
        self.g1.stress_loc = voigt2m_dev(x[:5])
        self.g1.stress_glob = self.g1.Q.T  @ self.g1.stress_loc @ self.g1.Q
        self.g2.stress_loc = voigt2m_dev(x[5:])
        self.g2.stress_glob = self.g2.Q.T  @ self.g2.stress_loc @ self.g2.Q
         
        if N1+N2 != 8:
            print(['Sum of the active slip sys in cluster higher than 8!'])
                 

    def solveambSVD_coupled(self, Dp):

        Dg1_relaxed = Dp - crystal_properties.symR1*self.g1.relaxation_sliprates[0] \
                         - crystal_properties.symR2*self.g1.relaxation_sliprates[1] 
        Dg2_relaxed = Dp - crystal_properties.symR1*self.g2.relaxation_sliprates[0] \
                         - crystal_properties.symR2*self.g2.relaxation_sliprates[1]
        Dg1_relaxed = self.g1.Qb.T @ Dg1_relaxed @ self.g1.Qb
        Dg2_relaxed = self.g2.Qb.T @ Dg2_relaxed @ self.g2.Qb
        Dp1_vec = m2voigt_dev(Dg1_relaxed)
        Dp2_vec = m2voigt_dev(Dg2_relaxed)
        Dp_vec = np.append(Dp1_vec, Dp2_vec)
        
        rss_g1 = getRSS(self.g1.stress_loc, crystal_properties.P3d)
        rss_g2 = getRSS(self.g2.stress_loc, crystal_properties.P3d)
        potentactivesID_g1 = np.where(rss_g1/self.g1.crss > (1.-Stol))[0]
        potentactivesID_g2 = np.where(rss_g2/self.g2.crss > (1.-Stol))[0]
        Nact_g1 = len(potentactivesID_g1)
        Nact_g2 = len(potentactivesID_g2)
        
        found_SVD_solution = False
        while not found_SVD_solution:         
            A = np.zeros((10, Nact_g1 + Nact_g2))
            for i, pID in enumerate(potentactivesID_g1):
                A[:5,i] = [crystal_properties.P[0,pID], 
                           crystal_properties.P[1,pID], 
                           crystal_properties.P[2,pID], 
                           crystal_properties.P[3,pID],
                           crystal_properties.P[4,pID]]
                
            for i, pID in enumerate(potentactivesID_g2):
                A[5:,Nact_g1+i] = [crystal_properties.P[0,pID], 
                                   crystal_properties.P[1,pID], 
                                   crystal_properties.P[2,pID], 
                                   crystal_properties.P[3,pID],
                                   crystal_properties.P[4,pID]]

            # compute pseudo-inverse of A, by SVD
            Apinv = np.linalg.pinv(A)
            gm_PA = np.dot(Apinv, Dp_vec)
            gm_g1 = np.zeros(24)
            gm_g2 = np.zeros(24)
            gm_g1[potentactivesID_g1] = gm_PA[:Nact_g1]
            gm_g2[potentactivesID_g2] = gm_PA[Nact_g1:]
                         
            # to be further testet whether it is a valid solution
            # check if gm_g1 is a valid solution
            tmpD = np.zeros((3,3))
            for i in potentactivesID_g1:
                tmpD += crystal_properties.P3d[:,:,i]*gm_g1[i]
            tmpdvec = m2voigt_dev(tmpD)
            maxdiff = np.max(np.abs(Dp1_vec - tmpdvec))
            if maxdiff > 1e-3:
                print('SVD problem grain 1')
                
                         
            # check if gm_g2 is a valid solution
            tmpD = np.zeros((3,3))
            for i in potentactivesID_g2:
                tmpD += crystal_properties.P3d[:,:,i]*gm_g2[i]
            tmpdvec = m2voigt_dev(tmpD)
            maxdiff = np.max(np.abs(Dp2_vec - tmpdvec))
            if maxdiff > 1e-3:
                print('SVD problem grain 2')

            if np.any(gm_PA < -Dtol):
                negative_ind = np.argmin(gm_PA)
                # remove the index of the most negative gm_PA from the set of indeces
                # of potentially active slip systems
                if negative_ind < Nact_g1:
                    potentactivesID_g1 = np.delete(potentactivesID_g1, negative_ind)
                    Nact_g1 -= 1
                else:
                    potentactivesID_g2 = np.delete(potentactivesID_g2, negative_ind-Nact_g1)
                    Nact_g2 -= 1
            else:
                found_SVD_solution = True

            # neither complete set of the basis solutions nor weights are caluclated
                
        self.g1.sliprates = gm_g1
        self.g2.sliprates = gm_g2                
                         
                        
# definition of the Grain_boundary class
class Grain_boundary:
    def __init__(self, euler_angles):
        self.init_euler_angles    = euler_angles
        self.Q0                   = ang2matrix(euler_angles)
        self.Q                    = self.Q0.copy()
        
    def updateOrientation(self, F):       
        # initializing the ellipsoid as a sphere
        C0 = np.eye(3)
        invF = np.linalg.inv(F)
        # get deformed ellipsoid
        C = (invF).T @ C0 @ invF
        # get axes x y z of coord system given by QB matrix
        vec10 = self.Q0[0,:]
        vec20 = self.Q0[1,:]
        vec30 = self.Q0[2,:]

        # apply deformation on a planar interface given by x and y axes, and
        # obtain normal axis z and after renormalizing and
        # reorthonormalizing obtain new QB matrix
        vec1 = F @ vec10
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = F @ vec20
        vec3 = np.cross(vec1,vec2)
        vec3 = vec3/np.linalg.norm(vec3)
        vec2 = np.cross(vec3,vec1)
        vec2 = vec2/np.linalg.norm(vec2)
        self.Q[0,:] = vec1
        self.Q[1,:] = vec2
        self.Q[2,:] = vec3
        

# definition of the Results class
class Results:
    def __init__(self):
        self.stress_loc           = None
        self.stress_glob          = None
        self.euler_angles         = None
        self.crss                 = None
        self.sliprates            = None
        self.activesID            = None
        self.num_actives          = None
        self.weights              = None
        self.stress_corners       = None
        self.slip_basis_solutions = None
        self.relaxation_sliprates = None
        self.statevar             = None


# definition of the Grain class
class Grain:
    def __init__(self, euler_angles, hardening_law=None):
        # initialize hardening variables
        if hardening_law is None:
            hardening_law = {'model':'RIGID_PLASTIC',    
                             'hardening_parameters':[10.],
                             'relax_penalties': [0.,0.,0.]}
        elif not 'hardening_parameters' in hardening_law.keys():
            hardening_law['hardening_parameters'] = [10.]
        elif not 'relax_penalties' in hardening_law.keys():
            hardening_law['relax_penalties'] = [0., 0., 0.]
        
        self.hardening_law        = hardening_law
        self.init_euler_angles    = euler_angles
        self.Q0                   = ang2matrix(euler_angles)
        self.Q                    = self.Q0
        self.R                    = np.eye(3)
        self.Dp                   = None              # plastic strain rate tensor expressend in grain's coord sys
        self.Dp_vec               = None              # 'Dp' saved as vector of length 5 (voigt notation and trace(Dp)=0)
        self.stress_loc           = None              # deviatoric Cauchy stress in grain coordinate system
        self.stress_glob          = None              # deviatoric Cauchy stress in global coordinate system
        self.sliprates            = np.zeros(24)
        self.relaxation_sliprates = np.zeros(2)
        self.activesID            = None
        self.total_sliprate       = 0.
        self.total_slip           = 0.
        self.w                    = None
        self.S_corners            = None
        self.slip_basis_solutions = None
        
        # saving results
        self.results = Results()
        
        hparams = self.hardening_law['hardening_parameters']
        if self.hardening_law['model'] == 'RIGID_PLASTIC':
            self.hvars = [hparams[0]]
            self.crss = np.full(24, hparams[0])
            
        elif self.hardening_law['model'] == 'BAUSCHINGER':
            self.hvars      = np.zeros(97)
            self.hvars[0]   = hparams[0]     # tau0
            self.hvars[73:] = hparams[0]     # initially crss equals tau0
            # total critical resolved shear stress as sum of all contributions
            self.crss = self.hvars[73:]      
            
        # initialize relaxation penalty stress for Alamel model
        self.relax_penalties = hardening_law['relax_penalties']
        
    def hardening(self, dt):
        hparams = hardening_law['hardening_parameters']
        if self.hardening_law['model'] == 'BAUSCHINGER':
            tau0, th2, th3, th4, gm2, gm3, qP, qL, qR, qLR, gmP, gmL, gmR = hparams
            tauI, tauL, tauP, tauR = self.hvars[0], self.hvars[1:25], self.hvars[25:49], self.hvars[49:73],
            reverseID   = self.activesID + 12*(1-2*(self.activesID//12))
            rev_actID   = np.concatenate((self.activesID, reverseID))  
            passivesID  = np.array(list(set(range(24))-set(rev_actID)))

            dGM = self.total_sliprate*dt

            tauLsat, tauPsat, tauRsat = np.zeros(24), np.zeros(24), np.zeros(24)
            tauLsat[passivesID]      =  qL*tauI
            tauPsat[self.activesID]  =  qP*tauI
            tauRsat[reverseID]       = -qR*tauI
            
            if self.hardening_law['algorithm'] == 'explicit':
                hL = (tauLsat - tauL)/gmL
                hP = (tauPsat - tauP)/gmP
                hR = (tauRsat - tauR)/gmR
                tauL += hL*dGM   # latent hardening
                tauP += hP*dGM   # polarisation
                tauR += hR*dGM   # reverse hardening - BAUSCHINGER effect
            elif self.hardening_law['algorithm'] == 'implicit':
                tauL = (1./(gmL+dGM))*(tauLsat*dGM + tauL*gmL)
                tauP = (1./(gmP+dGM))*(tauPsat*dGM + tauP*gmP)
                tauR = (1./(gmR+dGM))*(tauRsat*dGM + tauR*gmR)

            # isotropic hardening (extended Voce law)
            tauI = tau0 + th2*gm2*(1.-np.exp(-self.total_slip/gm2)) + th3*gm3*(1.-np.exp(-self.total_slip/gm3)) + th4*self.total_slip

            # update hardening variables
            self.hvars[0]     = tauI
            self.hvars[1:25]  = tauL
            self.hvars[25:49] = tauP
            self.hvars[49:73] = tauR
            self.hvars[73:]   = tauI + tauL + tauP + tauR
            # total critical resolved shear stress as sum of all contributions
            self.crss = self.hvars[73:]
        else:
            # here to define user hardening law
            pass
            
    def init_results_output(self, result_vars):
        
        Nout = result_vars['number_of_outputs']
        if result_vars['grain_results'] is not []:
            if 'stress_loc' in result_vars['grain_results']:
                self.results.stress_loc           = np.zeros((3,3,Nout))
            if 'stress_glob' in result_vars['grain_results']:
                self.results.stress_glob          = np.zeros((3,3,Nout))
            if 'euler_angles' in result_vars['grain_results']:
                self.results.euler_angles         = np.zeros((3,Nout)) 
            if 'crss' in result_vars['grain_results']:
                self.results.crss                 = np.zeros((24,Nout))
            if 'sliprates' in result_vars['grain_results']:
                self.results.sliprates            = np.zeros((24,Nout))
            if 'total_sliprate' in result_vars['grain_results']:
                self.results.total_sliprate       = np.zeros(Nout)
            if 'activesID' in result_vars['grain_results']:
                self.results.activesID            = np.zeros((24,Nout))
            if 'num_actives' in result_vars['grain_results']:
                self.results.num_actives          = np.zeros(Nout)
            if 'weights' in result_vars['grain_results']:
                self.results.weights              = []
            if 'stress_corners' in result_vars['grain_results']:
                self.results.stress_corners       = []
            if 'slip_basis_solutions' in result_vars['grain_results']:
                self.results.slip_basis_solutions = []
            if 'relaxation' in result_vars['grain_results']:
                self.results.relaxation_sliprates = np.zeros((2,Nout))
            if 'statevar' in result_vars['grain_results']:
                if self.hardening_law['model'] == 'BAUSCHINGER':
                    self.results.statevar = {
                         'tauI' : np.zeros(Nout),
                         'tauL' : np.zeros(24,Nout),
                         'tauP' : np.zeros(24,Nout),
                         'tauR' : np.zeros(24,Nout)
                                              } 
    
    def save_results(self, result_vars, outIND):
        if result_vars['grain_results'] is not []:
            if 'stress_loc' in result_vars['grain_results']:
                self.results.stress_loc[:,:,outIND] = self.stress_loc
            if 'stress_glob' in result_vars['grain_results']:
                self.results.stress_glob[:,:,outIND] = self.stress_glob
            if 'euler_angles' in result_vars['grain_results']:  
                self.results.euler_angles[:,outIND] = matrix2ang(self.Q) 
            if 'crss' in result_vars['grain_results']:
                self.results.crss[:,outIND] = self.crss
            if 'sliprates' in result_vars['grain_results']:
                self.results.sliprates[:,outIND] = self.sliprates
            if 'total_sliprate' in result_vars['grain_results']:
                self.results.total_sliprate[outIND] = np.sum(self.sliprates)
            if 'activesID' in result_vars['grain_results']:
                self.results.activesID[:len(self.activesID),outIND] = self.activesID
            if 'num_actives' in result_vars['grain_results']:
                self.results.num_actives[outIND] = len(self.activesID)
            if 'weights' in result_vars['grain_results']:
                self.results.weights.append(self.w)
            if 'stress_corners' in result_vars['grain_results']:
                self.results.stress_corners.append(self.S_corners)
            if 'slip_basis_solutions' in result_vars['grain_results']:
                self.results.slip_basis_solutions.append(self.slip_basis_solutions)
            if 'relaxation' in result_vars['grain_results']:
                self.results.relaxation_sliprates[:,outIND] = self.relaxation_sliprates
            if 'statevar' in result_vars['grain_results']:
                if self.hardening_law['model'] == 'BAUSCHINGER':
                    self.results.statevar['tauI'][outIND] = self.hvars[0]
                    self.results.statevar['tauL'][outIND] = self.hvars[1:25]
                    self.results.statevar['tauP'][outIND] = self.hvars[25:49]
                    self.results.statevar['tauR'][outIND] = self.hvars[49:73]
    



    def solveSingleCrystal(self, result_vars, solve_Tayloramb='SVD', SRS=0.01, gm0=1.):
        
        def funcRD(x, m, gm0):
            S = voigt2m_dev(x)
            tau = getRSS(S, crystal_properties.P3d)  
            tau[np.where(tau < 0)[0]] = 0.
            gm = gm0*(tau/self.crss)**(1./m)
            dvec = crystal_properties.P @ gm
            return self.Dp_vec - dvec

        # **************** RATE INDEPENDENT FULL CONSTRAINED TAYLOR MODEL *********     
        # For stress determination, cases with unique stress solutions  
        # (with at least 5 active ss in basis solution) are treated separatelly 
        # from degenerated cases witch no unique stress solution (less than 5
        # active ss in the basis solutions 
        # (Please check https://doi.org/10.1016/j.ijplas.2013.10.002)
        
        # run simplex method (linear programming) in order to find one slip-rate solution that
        # minimizes the internal energy and fullfills the contraints given by the macroscopic
        # plastic strain rate tensor
        self.findslips()

        # This is the main switch in the RI Taylor model in order to calculate stress. 
        # Solution gm found by simplex method with less then 5 active slip systems 
        # leads to non-corner stress solution, often called degeneracy in stress or 
        # stress ambiguity and needs special treatment (function "stress_corners"). 
        # If simplex method found 5 non-zero slip rates, function "getstress" 
        # calculates stress tensor from CRSS on each active slip systems 
        if len(self.activesID) == 5:                  
            # calculate the stress in grain's local coord sys
            self.stress_loc = getstress(self.activesID, self.crss)
            self.S_corners = [self.stress_loc]
        elif len(self.activesID) < 5:
            # the algorithm will find all the stress corners (stress_solutions='all', default)
            # or just one of the stress corners (stress_solutions='onecorner')
            self.get_stress_corners()
            # in case of stress degenerecy, calculate the average of the stress corners
            self.stress_loc = stress_average(self.S_corners)
                  

        # solving the Taylor ambiguity

        # In case of the Taylor ambiguity present, gm_corners will find 
        # either all (option gm_solutions='all') or set of linearly independent
        # basis solutions (option gm_solutions='indep')
        if solve_Tayloramb in ['QP', 'average'] or 'slip_basis_solutions' in result_vars['grain_results']:
            self.get_slip_corners()
        
        if solve_Tayloramb == 'QP':
            # solving ambiguity by quadratic programming - linearly 
            # indpendent set of basis solutions is needed. If not provided, 'solveamb_QP'
            # will calculate four linear independent basis solutions)
            self.sliprates, self.w = solveamb_QP(self.slip_basis_solutions)  
            
        elif solve_Tayloramb == 'SVD':
            # solving ambiguity by singular value decomposition (equal to QP but much faster!),
            self.solveambSVD()
            
        elif solve_Tayloramb == 'average':
            self.sliprates = np.average(self.slip_basis_solutions, axis=1)
            self.w = np.full(self.slip_basis_solutions.shape[1], 1./self.slip_basis_solutions.shape[1])
        
        elif solve_Tayloramb == 'RD':            
            gm0 = np.mean(self.sliprates)
            x0 = m2voigt_dev(self.stress_loc)
#             sol = scipy.optimize.root(funcRD, x0=x0, args=(SRS, gm0), method='hybr', options={'xtol':1e-5, 'maxfev':1000, 'epsfcn':1e-3, 'factor':100.})
            sol = scipy.optimize.root(funcRD, x0=x0, args=(SRS, gm0), method='krylov', options={'xtol':1e-5, 'ftol':1e-5, 'maxiter':1000,})
            if not sol.success:
                print('Residuals:')
                print(sol.fun)
                print(sol.message)         
            self.stress_loc = voigt2m_dev(sol.x)   
            tau = getRSS(self.stress_loc, crystal_properties.P3d)     
            tau[np.where(tau < 0)[0]] = 0.
            self.sliprates = gm0*(tau/self.crss)**(1./SRS)
            
        # update the IDs of the active slip systems after solving the Taylor ambiguity 
        self.activesID = np.where(self.sliprates > Dtol)[0]
        # total slip rate in grain
        self.total_sliprate = np.sum(self.sliprates)
    
    def findslips(self):
        
        sol = scipy.optimize.linprog(self.crss, A_ub=None, b_ub=None, A_eq=crystal_properties.P, 
                                     b_eq=self.Dp_vec, bounds=None, method='revised simplex', callback=None,
                                     options=None)   # bounds=None means by default positive solution only

        if sol.status != 0: # simplex failed to find a solution
            print(sol.message)
            print('Trying interior-point algorithm...')
            sol = scipy.optimize.linprog(f, A_ub=None, b_ub=None, A_eq=A_eq, 
                                     b_eq=b_eq, bounds=None, method='interior-point', callback=None)
            print(sol.message)

        # vector of 24 elements containing slip rates (all positive) that minimizes the internal energy
        self.sliprates = sol.x
              
        # array of indeces of active slip systems (signed). can by of length 1, 2, 3, 4 or 5
        self.activesID = np.where(self.sliprates > Dtol)[0]
    
        
    
    
    def get_stress_corners(self, stress_solutions='all'):
        '''
        Function stress_corners finds (if stress_solutions=='all') all the stress corner solutions
        of a degenerated case. Option stress_solutions=='onecorner' returns the first found stress solution,
        regardless of the total number of corner solutions.
        '''
        self.S_corners = []
        for ind in crystal_properties.sslookup:
            # ss_ind contains 5 active and "compatible" slipsystems
            if set(self.activesID).issubset(set(ind)):
                lind = list(ind)
                S = getstress(lind, self.crss)
                rss = getRSS(S, crystal_properties.P3d)
                # check whether rss have the same signs for all ss
                # and check whether the yield criterion is obeyed
                if np.all(rss[lind] >= 0.0) and (np.all(np.abs(rss)/self.crss <= (1. + Stol))):
                    already_in = False
                    if self.S_corners == []:
                        self.S_corners.append(S)
                        if stress_solutions == 'onecorner': return S
                    else:
                        for _S in self.S_corners:
                            if np.allclose(S, _S, atol=Stol):
                                already_in = True
                                break

                        if not already_in: 
                            self.S_corners.append(S)
      
    
    def get_slip_corners(self, gm_solutions='all'):
    
    # if solution = 'all' , all the corner slip solutions will be returned
    # if solution = 'indep', only linear independent set of max 4 slip solutions wll be returned

        try:
            S = self.S_corners[0]
        except:
            S = self.S_corners
        rss = getRSS(S, crystal_properties.P3d)
        potentactivesID = np.where(rss/self.crss > (1.-Stol))[0]
        sliprates_list, actives_list = np.zeros((24,12)), np.zeros((24,12))
        sliprates_list[:,0] = self.sliprates
        actives_list[:,0] = self.sliprates > Dtol
        # slip basis solution counter
        counter = 1
        # isdep variable controls linear dependence of columns in gm_list
        isdep   = False

        for ind in itertools.combinations(potentactivesID, 5):
            if ind in crystal_properties.sslookup:
                already_in = False
                valid = False

                # finds the stress corner corresponding to 5 slipsystems ind  
                row = crystal_properties.sslookup[ind]
                ind = list(ind)
                A = crystal_properties.Dcalc_lookup[row].reshape(5,5)
                gm = np.linalg.solve(A, self.Dp_vec)
                S = getstress(ind, self.crss)
                tau = getRSS(S, crystal_properties.P3d)

                # check whether tau and gm have the same signs
                if np.all(gm >= -Dtol) and np.all(tau[ind] >= -Stol):
                    gm_candid = np.zeros(24)
                    gm_candid[ind] = gm
                    slips_candid = gm_candid > Dtol
                    valid = True

                # gm is a valid solution, now it remains 
                # to check whether this solution has not already been found
                if valid:
                    for _slip in actives_list.T:  
                        if np.all(slips_candid == _slip): 
                            already_in = True
                            break
                            
                    if not already_in:
                        sliprates_list[:,counter] = gm_candid
                        actives_list[:,counter]   = slips_candid
                        counter += 1

                        if gm_solutions == 'indep' and counter >= 4:
                            if np.linalg.matrix_rank(sliprates_list, tol=None) == 4:
                                break # we found all the linear independent solutions
                            else:
                                isdep = True

        # remove the unfilled columns and rows in sliprates_list and actives_list
        sliprates_list = sliprates_list[:,:counter]
        actives_list   = actives_list[:,:counter]

        # remove the linear dependent vectors
        if isdep:
            ncols = sliprates_list.shape[1]
            for trycol in range(ncols):
                cols = [x for x in range(ncols) if x!=trycol]
                if np.linalg.matrix_rank(sliprates_list[:,cols]) == 4:
                    sliprates_list = sliprates_list[:,cols]
                    break

        self.slip_basis_solutions = sliprates_list
    
    
    def solveambSVD(self):

        rss = getRSS(self.stress_loc, crystal_properties.P3d)
        potentactivesID = np.where(rss/self.crss > (1.-Stol))[0]

        found_SVD_solution = False
        while not found_SVD_solution:

            A = crystal_properties.P[:,potentactivesID]

            # compute pseudo-inverse of A, by SVD
            Apinv = np.linalg.pinv(A)
            gm_PA = np.dot(Apinv, self.Dp_vec)
            gm = np.zeros(24)
            gm[potentactivesID] = gm_PA
            # to be further testet whether it is a valid solution

            # caluclate plastic strain rate vector in voigt notation from 'gm_PA'
            dvec = np.dot(crystal_properties.P, gm)

            # calculate the difference #IT IS JUST A CHECK AND CAN BE REMOVED TO SAVE COMPUTATIONS
            maxdiff = np.max(np.abs(self.Dp_vec - dvec))
            if maxdiff > 1e-3:
                print('SVD problem')

            if np.any(gm_PA < -Dtol):
                # remove the index of the most negative gm_PA from the set of indeces
                # of potentially active slip systems
                potentactivesID = np.delete(potentactivesID, np.where(gm_PA == min(gm_PA))[0][0])
            else:
                found_SVD_solution = True

            # neither complete set of the basis solutions nor weights are caluclated
                
        self.sliprates = gm
    

    def plot_orientation(self, plot_type='IPF', plot_border=True, plot_what='end', arrow_width=0.005, color='b', marker='.', markersize='1'):
        if plot_type == 'PF':
            if plot_border:
                # plot circle
                a1 = np.arange(0,2*np.pi,0.01)
                plt.plot(np.cos(a1), np.sin(a1),'k')
            
            planes111 = 1./np.sqrt(3.)*np.array([[ 1., 1., 1.],
                                                 [-1., 1., 1.],
                                                 [ 1.,-1., 1.],
                                                 [-1.,-1., 1.]])
                                                
            planes111rot = np.dot(self.Q.T, planes111.T)
            # projection of axis vector to the circle plane to get stereographic projection
            proj111 = np.zeros((2,4))
            for i in range(4):
                planes111rot[:,i] = np.sign(planes111rot[2,i])*planes111rot[:,i]
                proj111[0,i] = -planes111rot[1,i]/(planes111rot[2,i]+1)      
                proj111[1,i] =  planes111rot[0,i]/(planes111rot[2,i]+1)
            plt.plot(proj111[0,:], proj111[1,:], color=color, marker=marker, markersize=markersize, linestyle='')
        elif plot_type == 'IPF':
            x3 = 1./np.sqrt(3.)/(1./np.sqrt(3.)+1.)
            x2 = 1./np.sqrt(2.)/(1./np.sqrt(2.)+1.)
            
            if plot_border:
                # plot the triangle
                plt.plot([0., x2], [0., 0.],'k')
                plt.plot([0., x3], [0., x3],'k')
                plt.axis([-0.05, 0.5, -0.05, 0.5])

                # plot arc
                a1 = np.arange(0., 0.263, 0.001)
                plt.plot((1.+x2)*np.cos(a1)-1.,(1.+x2)*np.sin(a1),'k')
            
            if plot_what == 'trajectory':
                try:
                    shp = self.results.euler_angles.shape
                    if shp[1] > 1:
                        noresults = False
                        plot_data = np.zeros((2,shp[1]))
                        for i in range(shp[1]):
                            Q = ang2matrix(self.results.euler_angles[:,i])
                            plot_data[:,i] = ori2IPF(Q, [0.,0.,1.], (x2, x3))
                        plt.plot(plot_data[0,:], plot_data[1,:], color=color, marker=marker, markersize=markersize, ls='-', alpha=0.5)
                    else:
                        print('No orientation data available for plotting trajectories. Increase "number_of_outputs".')
                        noresults = True
                except:
                    print('No orientation data available for plotting trajectories. Include "euler_angles" to grain_results.')
                    noresults = True
                    
                if noresults:
                    # plot at least start and end orientation
                    plot_data = np.zeros((2,2))
                    plot_data[:,0] = ori2IPF(self.Q0, [0.,0.,1.], (x2, x3))
                    plot_data[:,1] = ori2IPF(self.Q, [0.,0.,1.], (x2, x3))
                    dx = plot_data[0,1] - plot_data[0,0]
                    dy = plot_data[1,1] - plot_data[1,0]
                    plt.arrow(plot_data[0,0], plot_data[1,0], dx, dy, width=arrow_width, length_includes_head=True)
                    
            elif plot_what == 'start_end':
                # plot at least start and end orientation
                plot_data = np.zeros((2,2))
                plot_data[:,0] = ori2IPF(self.Q0, [0.,0.,1.], (x2, x3))
                plot_data[:,1] = ori2IPF(self.Q, [0.,0.,1.], (x2, x3))
                dx = plot_data[0,1] - plot_data[0,0]
                dy = plot_data[1,1] - plot_data[1,0]
                plt.arrow(plot_data[0,0], plot_data[1,0], dx, dy, width=arrow_width, length_includes_head=True)
            elif plot_what == 'end':
                # typical case when plotting whole polycrystal
                xcoord, ycoord = ori2IPF(self.Q, [0.,0.,1.], (x2, x3))
                plt.plot(xcoord, ycoord, color=color, marker=marker, markersize=markersize, alpha=1)
        
        plt.gca().set_aspect('equal', adjustable='box')
        
    
    
    def yield_locus(self, number_of_points=None, plot_axes=None, exponent=20.):

        def funcYL(k, Sdir, Sabs, Ptmp, n, crss):
            S = k*Sdir + Sabs
            xi = np.ones(24)
            tmp = np.zeros(24)
            for i in range(24):
                tmp[i] = max(0., np.tensordot(S, Ptmp[:,:,i], axes=2)/crss[i])
            maxtmp = np.max(tmp)
            tmp /= maxtmp
            y = np.sum(xi*tmp**n)
            y = maxtmp*y**(1./n) - 1.
            return y

        Pglob = np.zeros((3,3,24))
        for i in range(24):
            Pglob[:,:,i] = self.Q.T @ crystal_properties.P3d[:,:,i] @ self.Q

        normal_components = [(0,0),(1,1),(2,2)]
        shear_components  = [(1,2),(0,2),(0,1)]

        xij = [int(x)-1 for x in plot_axes[0]]
        xij.sort()
        xij = tuple(xij)
        yij = [int(x)-1 for x in plot_axes[1]]
        yij.sort()
        yij = tuple(yij)
        xi, xj = xij
        yi, yj = yij
        if len(plot_axes) == 4:
            zij = [int(x)-1 for x in plot_axes[2]]
            zij.sort()
            zij = tuple(zij)
            zi, zj = zij
            if zij in [xij, yij]:
                sys.exit('Out-of-plane stress component must differ.')
            elif zij not in normal_components + shear_components:
                sys.exit('Out-of-plane stress component is NA.')
            s0_list = plot_axes[3]
        else:
            zij = None
            s0_list = [0.]

        wProg = ipywidgets.IntProgress(min=0, max=number_of_points*len(s0_list), description='Running:',
        bar_style='', # 'success', 'info', 'warning', 'danger' or ''
        orientation='horizontal')
        display(wProg)    

        angles = np.linspace(0, 2.*np.pi, number_of_points)
        YL = np.zeros((3,3,number_of_points*len(s0_list)))
        non_converg_stress = 0

        for s0i, s0 in enumerate(s0_list):
            for ind, ang in enumerate(angles):
                wProg.value += 1
#                 print('')
#                 print('{}: angle {}'.format(ind, np.rad2deg(ang)))
#                 print('')
                Sdir = np.zeros((3,3))
                Sabs = np.zeros((3,3))
                ind_Sdir = np.full((3, 3), False, dtype=bool)
                ind_Sabs = np.full((3, 3), True, dtype=bool)

                if xij in shear_components: # xij is a shear component
                    if yij in shear_components: # both are shear stress components
                        Sdir[xi, xj], Sdir[xj, xi] = np.cos(ang), np.cos(ang)
                        Sdir[yi, yj], Sdir[yj, yi] = np.sin(ang), np.sin(ang)
                        ind_Sdir[xi, xj], ind_Sdir[xj, xi] = True, True
                        ind_Sdir[yi, yj], ind_Sdir[yj, yi] = True, True
                        ind_Sabs[xi, xj], ind_Sabs[xj, xi] = False, False
                        ind_Sabs[yi, yj], ind_Sabs[yj, yi] = False, False
                        if zij in normal_components:
                            Sabs[zi,zj] = 2./3.*s0
                            rest_normal = [normal for normal in normal_components if normal != zij]
                            for rn in rest_normal:
                                Sabs[rn[0], rn[1]] = -1./3.*s0
                        elif zij in shear_components:
                            Sabs[zi,zj] = s0
                            Sabs[zj,zi] = s0
                        elif zij is not None:
                            sys.exit('Yield surface plotting problem...')

                    else: # yij is a normal component
                        Sdir[xi, xj] = np.cos(ang)
                        Sdir[xj, xi] = np.cos(ang)
                        Sdir[yi, yj] = 2./3.*np.sin(ang)
                        ind_Sdir[xi, xj], ind_Sdir[xj, xi] = True, True
                        ind_Sdir[yi, yj] = True
                        ind_Sabs[xi, xj], ind_Sabs[xj, xi] = False, False
                        ind_Sdir[yi, yj] = False
                        rest_normal = [normal for normal in normal_components if normal != yij]
                        for rn in rest_normal:
                            Sdir[rn[0], rn[1]] = -1./3.*np.sin(ang)
                            ind_Sdir[rn[0], rn[1]] = True
                            ind_Sabs[rn[0], rn[1]] = False
                        if zij in normal_components:
                            print('Out-of-plane stress component cannot be a normal component for this plot.')
                        elif zij in shear_components:
                            Sabs[zi, zj] = s0
                            Sabs[zj, zi] = s0
                        elif zij is not None:
                            sys.exit('Yield surface plotting problem...')

                else: # xij is a normal component
                    if yij in shear_components: # yij is a shear component
                        Sdir[yi, yj] = np.sin(ang)
                        Sdir[yj, yi] = np.sin(ang)
                        Sdir[xi, xj] = 2./3.*np.cos(ang)
                        ind_Sdir[yi, yj], ind_Sdir[yj, yi] = True, True
                        ind_Sdir[xi, xj] = True
                        ind_Sabs[yi, yj], ind_Sabs[yj, yi] = False, False
                        ind_Sabs[xi, xj] = False
                        rest_normal = [normal for normal in normal_components if normal != xij]
                        for rn in rest_normal:
                            Sdir[rn[0], rn[1]] = -1./3.*np.cos(ang)
                            ind_Sdir[rn[0], rn[1]] = True
                            ind_Sabs[rn[0], rn[1]] = False
                        if zij in normal_components:
                            print('Out-of-plane stress component cannot be a normal component for this plot.')
                        elif zij in shear_components:
                            Sabs[zi, zj] = s0
                            Sabs[zj, zi] = s0
                        elif zij is not None:
                            sys.exit('Yield surface plotting problem...')

                    else: # both are normal stress components
                        rest_normal = [normal for normal in normal_components if normal not in [xij, yij]]
                        rest_normal = rest_normal[0] # extract the only tuple from the list
                        Sdir[xi, xj] = 2./3.*np.cos(ang) - 1./3.*np.sin(ang)
                        Sdir[yi, yj] = 2./3.*np.sin(ang) - 1./3.*np.cos(ang)
                        Sdir[rest_normal[0], rest_normal[1]] = -1./3.*np.cos(ang) - 1./3.*np.sin(ang)
                        ind_Sdir[0, 0], ind_Sdir[1, 1], ind_Sdir[2, 2] = True, True, True
                        ind_Sabs[0, 0], ind_Sabs[1, 1], ind_Sabs[2, 2] = False, False, False
                        if zij in normal_components:
                            print('Out-of-plane stress component cannot be a normal component for this plot.')
                        elif zij in shear_components:
                            Sabs[zi, zj] = s0
                            Sabs[zj, zi] = s0
                        elif zij is not None:
                            sys.exit('Yield surface plotting problem...')
                            
                # initial guess
                if ind == 0:
                    x0 = np.mean(self.crss)
                elif ier == 1:
                    x0 = x
                else:
                    x0 = np.mean(self.crss)
                            
                x, infodict, ier, mesg = scipy.optimize.fsolve(funcYL, x0=x0, args=(Sdir, Sabs, Pglob, exponent, self.crss), full_output=1, epsfcn=1e-5)
                if ier == 1:
                    YL[:,:,ind+number_of_points*s0i] = x*Sdir + Sabs
                else:
                    numiter = 0
                    while ier != 1 and numiter < 20:
                        x, infodict, ier, mesg = scipy.optimize.fsolve(funcYL, x0=np.mean(self.crss)*np.random.rand(), args=(Sdir, Sabs, Pglob, exponent, self.crss), full_output=1, epsfcn=1e-5)
                        numiter += 1
                    if ier == 1:
                        YL[:,:,ind+number_of_points*s0i] = x*Sdir + Sabs
                    else:
                        print('The stress point is not well converged and will be omitted.')
                        YL[:,:,ind+number_of_points*s0i] = np.full((3,3), np.nan)   
                        non_converg_stress += 1

                        
        rij = normal_components.copy()
        if xij in rij:
            rij.remove(xij)
        if yij in rij:
            rij.remove(yij)
        if len(rij) == 1:
            rij = 2*rij
        if len(rij) == 3:
            rij = None
        if rij is not None:
            tmp = 0.5*(YL[rij[0][0],rij[0][1],:] + YL[rij[1][0],rij[1][1],:])
            YL[0,0,:] -= tmp
            YL[1,1,:] -= tmp
            YL[2,2,:] -= tmp
        
        return YL, non_converg_stress
            
        
        
# definition of the Crystal class
class Crystallography:
    def __init__(self, crystal_structure):
        self.calc_slipsystem_matrices(crystal_structure)
            
    # initialize cryllographic slip systems, Schmidt matrix
    def calc_slipsystem_matrices(self, crystal_structure):

        if crystal_structure == 1:
            # slip systems described by vectors n and b
            #n(s,i)
            n = np.zeros((12,3))
            n[0,:]  = 3**(-0.5)*np.array([ 1,  1, -1])
            n[1,:]  = 3**(-0.5)*np.array([ 1,  1, -1])
            n[2,:]  = 3**(-0.5)*np.array([ 1,  1, -1])
            n[3,:]  = 3**(-0.5)*np.array([ 1, -1, -1])
            n[4,:]  = 3**(-0.5)*np.array([ 1, -1, -1])
            n[5,:]  = 3**(-0.5)*np.array([ 1, -1, -1])
            n[6,:]  = 3**(-0.5)*np.array([ 1, -1,  1])
            n[7,:]  = 3**(-0.5)*np.array([ 1, -1,  1])
            n[8,:]  = 3**(-0.5)*np.array([ 1, -1,  1])
            n[9,:]  = 3**(-0.5)*np.array([ 1,  1,  1])
            n[10,:] = 3**(-0.5)*np.array([ 1,  1,  1])
            n[11,:] = 3**(-0.5)*np.array([ 1,  1,  1])
            self.n = n
            
            #b(s,i)
            b = np.zeros((12,3))
            b[0,:]  = 2**(-0.5)*np.array([ 0,  1,  1])
            b[1,:]  = 2**(-0.5)*np.array([ 1,  0,  1])
            b[2,:]  = 2**(-0.5)*np.array([ 1, -1,  0])
            b[3,:]  = 2**(-0.5)*np.array([ 0,  1, -1])
            b[4,:]  = 2**(-0.5)*np.array([ 1,  0,  1])
            b[5,:]  = 2**(-0.5)*np.array([ 1,  1,  0])
            b[6,:]  = 2**(-0.5)*np.array([ 0,  1,  1])
            b[7,:]  = 2**(-0.5)*np.array([ 1,  0, -1])
            b[8,:]  = 2**(-0.5)*np.array([ 1,  1,  0])
            b[9,:]  = 2**(-0.5)*np.array([ 0,  1, -1])
            b[10,:] = 2**(-0.5)*np.array([ 1,  0, -1])
            b[11,:] = 2**(-0.5)*np.array([ 1, -1,  0])
            self.b = b
            
            # Schimd tensor
            _mm    = np.zeros((3,3,12))
            _P     = np.zeros((3,3,12))
            _Omega = np.zeros((3,3,12))
            for i in range(12):
                _mm[:,:,i]    = np.outer(b[i,:],n[i,:])
                _P[:,:,i]     = 0.5*(_mm[:,:,i] + _mm[:,:,i].T)    # P = sym(mm)
                _Omega[:,:,i] = 0.5*(_mm[:,:,i] - _mm[:,:,i].T)    # Omega = skw(mm)  
            
            P24     = np.concatenate((_P, -_P), axis=2)
            Omega24 = np.concatenate((_Omega, -_Omega), axis=2)
            self.P3d = P24
            
            # more effective storage of P and Omega since P is a symmetric matrix with trace(P) = 0 and 
            # Omega is a skewsymmetric matrix
            Omega = np.zeros((3,24))
            P     = np.zeros((5,24))
            for i in range(24):
                Omega[:,i] = [Omega24[1,2,i], Omega24[0,2,i], Omega24[0,1,i]]
                P[:,i]     = [P24[0,0,i], P24[1,1,i], P24[1,2,i], P24[0,2,i], P24[0,1,i]]

            # generate precalculated tables:
            # 
            # sslookup is dictionary with keys equal to 5ers of geometrically compatible slipsystems
            #                        and item is a list, where first item is counting index following
            #                        by the admissible signs of shears (obeying yield criterion)
            comb5ers = itertools.combinations(range(24), 5)
            self.sslookup = {}

            # generation of lookup matrices Dcalc_lookup and Scalc_lookup 
            # for all the 384 admissible combinations of 5 slip systems out of 12 slip systems
            self.Dcalc_lookup = np.zeros((12288,25))
            self.Scalc_lookup = np.zeros((12288,25))
            i = 0
            for ind in comb5ers:
                A = np.zeros((5,5))
                for j in range(5):
                    A[:,j]  = np.array([ P[0,ind[j]], P[1,ind[j]], P[2,ind[j]], P[3,ind[j]], P[4,ind[j]] ]) 

                if np.linalg.matrix_rank(A, tol=None) > 4:
                    self.sslookup[ind] = i
                    self.Dcalc_lookup[i,:] = np.reshape(A,-1)  # flattening the A matrix

                    # generate Scalc_lookup matrix
                    B = np.zeros((5,5))
                    for j in range(5):
                        B[j,:] = np.array([2.*P[0,ind[j]]+P[1,ind[j]], 
                                           2.*P[1,ind[j]]+P[0,ind[j]],
                                           2.*P[2,ind[j]], 
                                           2.*P[3,ind[j]],
                                           2.*P[4,ind[j]]])

                    self.Scalc_lookup[i,:] = np.reshape(B,-1)  # flattening the B matrix                                     
                    i += 1


            # generate matrix for projection of Cauchy stress into RSS
            # it is B-like matrix, just for all 12 slip systems
            self.S_proj = 2.*P.T
            tmp0 = P.T[:,0]
            tmp1 = P.T[:,1]
            self.S_proj[:,0] += tmp1
            self.S_proj[:,1] += tmp0
            self.Omega = Omega
            self.P = P
            
            # definition of relaxations R1 and R2. Grains in one cluster will have opposite relaxations
            self.R1, self.R2 = np.zeros((3,3)), np.zeros((3,3))
            self.R1[0,2] = 1.
            self.R2[1,2] = 1.

            self.symR1 = getsym(self.R1)
            self.symR2 = getsym(self.R2)
        
        else:
            pass
    
                    
# definition of the Polycrystal class
class Polycrystal:
    
    # tolerances for the stress and strain-rate calculation
    global Dtol, Stol
    Dtol, Stol = 1e-8, 1e-8        
    
    def __init__(self, crystal_structure=None, orientations=None, elasticity=None, hardening_law=None, grain_interaction=None):
        
        # total plastic deformation gradient
        self.Fp = np.eye(3)
        
        # checking the input validity
        if orientations is not None:
            self.orientations = orientations
        else:
            sys.exit('You need to specify orientations.')
            
        if crystal_structure.upper() in ('FCC_111','FCC_110','FCC_100','FCC_112','BCC_110'):
            self.crystal_structure = crystal_structure_dict[crystal_structure] 
        else:
            sys.exit('Crystal structure not recognized.')
            
        if elasticity is None or type(elasticity) is np.ndarray:
            self.elasticity = elasticity
        else:
            sys.exit('Elasticity definition not recognized.')
            
        if hardening_law is None or hardening_law['model'].upper() in ('RIGID_PLASTIC','BAUSCHINGER'):
            self.hardening_law = hardening_law
        else:
            sys.exit('Hardening law not recognized.')
            
        if grain_interaction is None:
            grain_interaction = 'FCTAYLOR'
        if grain_interaction.upper() in ('FCTAYLOR','ALAMEL','ALAMEL3'):  
            self.grain_interaction = grain_interaction.upper()
        else:
            sys.exit('Grain interaction not recognized.')
        
        global crystal_properties 
        crystal_properties = Crystallography(self.crystal_structure)        
        
        # saving results
        self.average_stress = None
        self.average_slip   = None
        self.result_vars    = {}
        self.output_steps   = []
        
        # assigning the orientations to grains and in case of Alamel models to grain boundaries
        self.read_input_orientations()
            
        # calculate elastic modulus
        self.elast_modulus()       
    
    # loading/deforming the polycrystal
    def load(self, L=None, S_direction=None, S_absolute=None, iL=None, iS_direction=None, iS_absolute=None,
             Nsteps=None, dt=None, run_elasticity=False, result_vars={}, dofortran=True, 
             options = {'increment_jacobian': 1.e-3, 
                       'rotate_boundary'    : True,
                       'use_SCYL'           : False,
                       'SCYL_exponent'      : 100,
                       'solve_Tayloramb'    : 'average',
                       'gmdot0'             : 1.,
                       'SRS'                : 0.01}):
           
        self.options = options
        # correct if more output steps than actual steps
        Nout = result_vars['number_of_outputs']
        Nout = min(Nout, Nsteps)   
        result_vars['number_of_outputs'] = Nout
        # calculate the output steps
        self.output_steps = np.linspace(Nsteps, 0, Nout, endpoint=False, dtype=np.int32)
        self.Nsteps = Nsteps
        self.dt = dt
            
        ### initializing the result variables
        # if some polycrystal results are required, do initialize those polycrystal result variables
        self.result_vars = result_vars
        
        if 'polycrystal_results' in result_vars.keys():
            if 'average_stress' in result_vars['polycrystal_results']:
                self.average_stress = np.zeros((3,3,Nout))
            if 'average_slip' in result_vars['polycrystal_results']:
                self.average_slip   = np.zeros(Nout)     
        elif 'grain_results' not in result_vars.keys():
            print('No result output is specified.\n \
                  The orientation matrices calculated in the last time step \
                  will be the only simulation result.')
        
        # if some grain results are required, do initialize those grain result variables
        if result_vars['grain_results'] is not []:
            if self.grain_interaction != 'FCTAYLOR':
                for cluster in self.clusters:
                    cluster.g1.init_results_output(result_vars)
                    cluster.g2.init_results_output(result_vars)
            else:
                for grain in self.grains:
                    grain.init_results_output(result_vars)
        
        ### load/deformation definition check
        # TODO remove Sabs from user interface, make it hidden
        if L is None:
            self.L = np.zeros((3,3))
        else:
            self.L = np.asarray(L)
            
        if S_absolute is None:
            self.Sabs = np.zeros((3,3))
        else:
            self.Sabs = np.asarray(S_absolute)
            
        if S_direction is None:
            self.Sdir = np.zeros((3,3))
        else:
            self.Sdir = np.asarray(S_direction)
            
        if type(iL) is bool:
            iL = np.full((3,3), iL, dtype=bool)
        self.iL = np.asarray(iL)
            
        if type(iS_direction) is bool:
            iS_direction = np.full((3,3), iS_direction, dtype=bool)
        self.iSdir = np.asarray(iS_direction)
        
        if type(iS_absolute) is bool:
            iS_absolute = np.full((3,3), iS_absolute, dtype=bool)
        self.iSabs = np.asarray(iS_absolute)
        
        iS = self.iSdir + self.iSabs
        
        self.iDdir = self.iL*self.iL.T
        self.Ddir  = getsym(self.L)
 
        if not np.all(iS == (iS+iS.T)):
            sys.exit('Boundary conditions for stress tensor are not symmetric.')
        
        # stress and velocity gradient boundary conditions must be mutualy exclusive, 
        # also if only one off-diagonal elements in L is prescribed
        if np.any(iL*iL.T == iS):
            sys.exit('Check the mixed boundary conditions. They are not mutualy exclusively defined.')
        
        if not ( (all(np.diag(iL)) is True and all(np.diag(iS)) is False) or (all(np.diag(iL)) is False and all(np.diag(iS)) is True) ):
            sys.exit('All or None diagonal terms must be prescribed in L or deviatoric stress tensor.')
                

        ### run crystal plasticity
        if self.grain_interaction == 'FCTAYLOR': 
            self.rotate_boundary = None
            if not run_elasticity:                                                      
                if dofortran and not ImportErrCpfort:
                    self.init_cpfort()
                    # initializing the global variables
                    cpfort.globals.dt                                      = self.dt
                    cpfort.globals.nsteps                                  = self.Nsteps
                    cpfort.globals.nout                                    = len(self.output_steps)
                    cpfort.globals.output_steps                            = np.zeros(len(cpfort.globals.output_steps)) 
                    cpfort.globals.output_steps[:len(self.output_steps)]   = self.output_steps
                    outputvars                                             = self.result_vars['polycrystal_results'] + self.result_vars['grain_results']
                    outputvarsfort                                         = [outputvars_dict[var] for var in outputvars_dict if var in outputvars]
                    cpfort.globals.outputvars[:len(outputvarsfort)]        = outputvarsfort
                    cpfort.globals.grain_interaction                       = grain_interaction_dict[self.grain_interaction]
                    # boundary condictions
                    cpfort.globals.ind_sdir                                = m2voigt_dev(self.iSdir)
                    cpfort.globals.ind_sabs                                = m2voigt_dev(self.iSabs)
                    cpfort.globals.ind_d                                   = m2voigt_dev(self.iDdir)
                    cpfort.globals.sdir_prescribed                         = m2voigt_dev(self.Sdir)
                    cpfort.globals.sabs_prescribed                         = m2voigt_dev(self.Sabs)
                    cpfort.globals.d_prescribed                            = m2voigt_dev(self.Ddir) # total prescribed strain-rate
                    cpfort.globals.w_prescribed                            = getskw(self.L) # total prescribed spin
                    ndim                                                   = len(cpfort.globals.ind_d[cpfort.globals.ind_d == False])
                    cpfort.globals.eps4jacobian                            = options['increment_jacobian']
                    cpfort.scylglobals.scylon                              = options['use_SCYL']
                    cpfort.scylglobals.scylexp                             = options['SCYL_exponent']

                    dguess = make_good_guess(self.Sabs, self.iSabs, self.Sdir, self.iSdir, cpfort.globals.crss_mean)
                    dguess = m2voigt_dev(dguess)
                    if ndim == 0:
                        ndim = 5
                    Dsolved, Ssolved, info, resid, nfev = cpfort.crystal_plasticity.taylor(dguess=dguess, ndim=ndim)
                    self.save_results_taylorfort()
                    cpfort.crystal_plasticity.dealloc_outputvars_taylor()
                else:
                    if np.all(iL):                                                          
                        # run the deformation-driven full-constrained RIGID plastic Taylor model
                        self.TaylorFC()
                    else:
                        # run the RIGID plastic Taylor model with mixed boundary conditions
                        sys.exit('Specify "dofortran=True" for mixed boundary conditions.')
            else:
                sys.exit('Elastic-plastic Taylor-Lin model is not yet implemented.')
                        
        elif self.grain_interaction in ['ALAMEL','ALAMEL3']:
            self.rotate_boundary = options['rotate_boundary']
            if dofortran and not ImportErrCpfort:
                self.init_cpfort()
                # initializing the global variables    
                cpfort.globals.relax_penalties                         = self.hardening_law['relax_penalties']
                cpfort.globals.dt                                      = self.dt
                cpfort.globals.nsteps                                  = self.Nsteps
                cpfort.globals.nout                                    = len(self.output_steps)
                cpfort.globals.output_steps                            = np.zeros(len(cpfort.globals.output_steps)) 
                cpfort.globals.output_steps[:len(self.output_steps)]   = self.output_steps
                self.output_steps
                outputvars                                             = self.result_vars['polycrystal_results'] + self.result_vars['grain_results']
                outputvarsfort                                         = [outputvars_dict[var] for var in outputvars_dict if var in outputvars]
                cpfort.globals.outputvars[:len(outputvarsfort)]        = outputvarsfort
                cpfort.globals.grain_interaction                       = grain_interaction_dict[self.grain_interaction]
                # boundary condictions
                cpfort.globals.ind_sdir                                = m2voigt_dev(self.iSdir)
                cpfort.globals.ind_sabs                                = m2voigt_dev(self.iSabs)
                cpfort.globals.ind_d                                   = m2voigt_dev(self.iDdir)
                cpfort.globals.sdir_prescribed                         = m2voigt_dev(self.Sdir)
                cpfort.globals.sabs_prescribed                         = m2voigt_dev(self.Sabs)
                cpfort.globals.d_prescribed                            = m2voigt_dev(self.Ddir) # total prescribed strain-rate
                cpfort.globals.w_prescribed                            = getskw(self.L) # total prescribed spin
                ndim                                                   = len(cpfort.globals.ind_d[cpfort.globals.ind_d == False])
                cpfort.globals.eps4jacobian                            = options['increment_jacobian']
                cpfort.globals.rotate_boundary                         = self.rotate_boundary
                
                dguess = make_good_guess(self.Sabs, self.iSabs, self.Sdir, self.iSdir, cpfort.globals.crss_mean)
                dguess = m2voigt_dev(dguess)
                
                if ndim == 0:
                    ndim = 5
                Dsolved, Ssolved, info, resid, nfev = cpfort.crystal_plasticity.alamel(dguess=dguess, ndim=ndim)
                self.save_results_alamelfort()
                cpfort.crystal_plasticity.dealloc_outputvars_alamel()
            else:
                if np.all(iL):
                    self.Alamel()
                else:
                    # run the RIGID plastic Alamel model in with mixed boundary conditions
                    sys.exit('Specify "dofortran=True" for mixed boundary conditions.')
                
        else:
            print('Choose one of the defined analysis types: FC_TAYLOR, ALAMEL or ALAMEL3')
            sys.exit()
      
    def init_cpfort(self):
        """Initialize definition of polycrystal for fortran implementation of ALAMEL""" 
        
        if self.grain_interaction == 'FCTAYLOR':
            Q_list = np.zeros((3,3,self.Ngrains))
            
            if self.hardening_law['model'] == 'RIGID_PLASTIC':
                inithvar_list = np.full((1,self.Ngrains), self.hardening_law['hardening_parameters'][0])
            elif self.hardening_law['model'] == 'BAUSCHINGER':
                inithvar_list = np.zeros((97,self.Ngrains))
            else:
                pass # other models
            
            for i, g in enumerate(self.grains):
                Q_list[:,:,i] = g.Q
                if self.hardening_law['model'] == 'BAUSCHINGER':
                    inithvar_list[:,i] = g.hvars

            cpfort.crystal_plasticity.init_slipsystems(self.crystal_structure)
            cpfort.crystal_plasticity.init_orientation_taylor(Q_list)
            cpfort.crystal_plasticity.init_hardening_taylor(inithvar_list, self.hardening_law['hardening_parameters'], 
                                                            hardening_model_dict[self.hardening_law['model']])
            
            
        elif self.grain_interaction in ['ALAMEL','ALAMEL3']:
            Qg1_list = np.zeros((3,3,self.Nclusters))
            Qg2_list = np.zeros((3,3,self.Nclusters))
            Qb_list  = np.zeros((3,3,self.Nclusters))
        
            if self.hardening_law['model'] == 'RIGID_PLASTIC':
                inithvarG1_list = np.full((1,self.Nclusters), self.hardening_law['hardening_parameters'][0])
                inithvarG2_list = np.full((1,self.Nclusters), self.hardening_law['hardening_parameters'][0])
            elif self.hardening_law['model'] == 'BAUSCHINGER':
                inithvarG1_list = np.zeros((97,self.Nclusters))
                inithvarG2_list = np.zeros((97,self.Nclusters))
            else:
                pass # other models

            for i, c in enumerate(self.clusters):
                Qg1_list[:,:,i] = c.g1.Q
                Qg2_list[:,:,i] = c.g2.Q
                Qb_list[:,:,i]  = c.gb.Q
                if self.hardening_law['model'] == 'BAUSCHINGER':
                    inithvarG1_list[:,i] = c.g1.hvars
                    inithvarG2_list[:,i] = c.g2.hvars

            cpfort.crystal_plasticity.init_slipsystems(self.crystal_structure)
            cpfort.crystal_plasticity.init_orientation_alamel(Qg1_list, Qg2_list, Qb_list)
            cpfort.crystal_plasticity.init_hardening_alamel(inithvarG1_list, inithvarG1_list, self.hardening_law['hardening_parameters'], 
                                                            hardening_model_dict[self.hardening_law['model']])
        
    def save_results_taylorfort(self):
        self.crss_mean = cpfort.globals.crss_mean
        # hand over the current (updated) orientations from F2Py
        for i, g in enumerate(self.grains):
            g.Q = cpfort.globals.qg1_list[:,:,i]
            if self.hardening_law['model'] == 'BAUSCHINGER':
                g.hvars = cpfort.globals.hvarq1_list[:,i]
        
        # hand over the polycrystal results from F2Py 
        if 'average_stress' in self.result_vars['polycrystal_results']:
            self.average_stress = voigt2m_dev(cpfort.globals.out_average_stress)
        if 'average_slip' in self.result_vars['polycrystal_results']:
            self.average_slip = cpfort.globals.out_average_slip
        
        # hand over the grain results from F2Py
        if self.result_vars['grain_results'] is not []:
            for i, g in enumerate(self.grains):
                if 'stress_loc' in self.result_vars['grain_results']:
                    g.results.stress_loc = voigt2m_dev(cpfort.globals.out_stress_locg1[:,i,:])
                if 'stress_glob' in self.result_vars['grain_results']:
                    g.results.stress_glob = voigt2m_dev(cpfort.globals.out_stress_globg1[:,i,:])
                if 'euler_angles' in self.result_vars['grain_results']:  
                    g.results.euler_angles = cpfort.globals.out_euler_anglesg1[:,i,:].copy()
                if 'crss' in self.result_vars['grain_results']:
                    g.results.crss = cpfort.globals.out_crssg1[:,i,:].copy()
                if 'sliprates' in self.result_vars['grain_results']:
                    g.results.sliprates = cpfort.globals.out_slipratesg1[:,i,:].copy()
                if 'total_sliprate' in self.result_vars['grain_results']:
                    g.results.total_sliprate = np.sum(cpfort.globals.out_slipratesg1[:,i,:], axis=0)
                if 'activesID' in self.result_vars['grain_results']:
                    g.results.activesID = cpfort.globals.out_activesidg1[:,i,:].copy()
                if 'num_actives' in self.result_vars['grain_results']:
                    c.g.results.num_actives = cpfort.globals.out_num_activesg1[i,:].copy()
                if 'statevar' in self.result_vars['grain_results']:
                    if self.hardening_law['model'] == 'BAUSCHINGER':
                        g.results.statevar['tauI'] = cpfort.globals.out_statevarg1[0,i,:].copy()
                        g.results.statevar['tauL'] = cpfort.globals.out_statevarg1[1:25,i,:].copy()
                        g.results.statevar['tauP'] = cpfort.globals.out_statevarg1[25:49,i,:].copy()
                        g.results.statevar['tauR'] = cpfort.globals.out_statevarg1[49:73,i,:].copy()
        
    def save_results_alamelfort(self):
        self.crss_mean = cpfort.globals.crss_mean
        # hand over the current (updated) orientations from F2Py
        for i, c in enumerate(self.clusters):
            c.g1.Q = cpfort.globals.qg1_list[:,:,i]
            c.g2.Q = cpfort.globals.qg2_list[:,:,i]
            c.gb.Q = cpfort.globals.qb_list[:,:,i]
            if self.hardening_law['model'] == 'BAUSCHINGER':
                c.g1.hvars = cpfort.globals.hvarq1_list[:,i]
                c.g2.hvars = cpfort.globals.hvarq2_list[:,i]
        
        # hand over the polycrystal results from F2Py 
        if 'average_stress' in self.result_vars['polycrystal_results']:
            self.average_stress = voigt2m_dev(cpfort.globals.out_average_stress)
        if 'average_slip' in self.result_vars['polycrystal_results']:
            self.average_slip = cpfort.globals.out_average_slip
        
        # hand over the grain results from F2Py
        if self.result_vars['grain_results'] is not []:
            for i, c in enumerate(self.clusters):
                if 'stress_loc' in self.result_vars['grain_results']:
                    c.g1.results.stress_loc = voigt2m_dev(cpfort.globals.out_stress_locg1[:,i,:])
                    c.g2.results.stress_loc = voigt2m_dev(cpfort.globals.out_stress_locg2[:,i,:])
                if 'stress_glob' in self.result_vars['grain_results']:
                    c.g1.results.stress_glob = voigt2m_dev(cpfort.globals.out_stress_globg1[:,i,:])
                    c.g2.results.stress_glob = voigt2m_dev(cpfort.globals.out_stress_globg2[:,i,:])
                if 'euler_angles' in self.result_vars['grain_results']:  
                    c.g1.results.euler_angles = cpfort.globals.out_euler_anglesg1[:,i,:].copy()
                    c.g2.results.euler_angles = cpfort.globals.out_euler_anglesg2[:,i,:].copy()
                if 'crss' in self.result_vars['grain_results']:
                    c.g1.results.crss = cpfort.globals.out_crssg1[:,i,:].copy()
                    c.g2.results.crss = cpfort.globals.out_crssg2[:,i,:].copy()
                if 'sliprates' in self.result_vars['grain_results']:
                    c.g1.results.sliprates = cpfort.globals.out_slipratesg1[:,i,:].copy()
                    c.g2.results.sliprates = cpfort.globals.out_slipratesg2[:,i,:].copy()
                if 'total_sliprate' in self.result_vars['grain_results']:
                    c.g1.results.total_sliprate = np.sum(cpfort.globals.out_slipratesg1[:,i,:], axis=0)
                    c.g2.results.total_sliprate = np.sum(cpfort.globals.out_slipratesg2[:,i,:], axis=0)
                if 'activesID' in self.result_vars['grain_results']:
                    c.g1.results.activesID = cpfort.globals.out_activesidg1[:,i,:].copy()
                    c.g2.results.activesID = cpfort.globals.out_activesidg2[:,i,:].copy()
                if 'num_actives' in self.result_vars['grain_results']:
                    c.g1.results.num_actives = cpfort.globals.out_num_activesg1[i,:].copy()
                    c.g2.results.num_actives = cpfort.globals.out_num_activesg2[i,:].copy()
                if 'statevar' in self.result_vars['grain_results']:
                    if self.hardening_law['model'] == 'BAUSCHINGER':
                        c.g1.results.statevar['tauI'] = cpfort.globals.out_statevarG1[0,i,:].copy()
                        c.g2.results.statevar['tauI'] = cpfort.globals.out_statevarG1[0,i,:].copy()
                        c.g1.results.statevar['tauL'] = cpfort.globals.out_statevarG1[1:25,i,:].copy()
                        c.g2.results.statevar['tauL'] = cpfort.globals.out_statevarG1[1:25,i,:].copy()
                        c.g1.results.statevar['tauP'] = cpfort.globals.out_statevarG1[25:49,i,:].copy()
                        c.g2.results.statevar['tauP'] = cpfort.globals.out_statevarG1[25:49,i,:].copy()
                        c.g1.results.statevar['tauR'] = cpfort.globals.out_statevarG1[49:73,i,:].copy()
                        c.g2.results.statevar['tauR'] = cpfort.globals.out_statevarG1[49:73,i,:].copy()
                if 'relaxation' in self.result_vars['grain_results']:
                    c.g1.results.relaxation_sliprates = cpfort.globals.out_relaxsliprates[:,i,:].copy()
                    c.g2.results.relaxation_sliprates = cpfort.globals.out_relaxsliprates[:,i,:].copy()
    

    
    def TaylorFC(self):
        wProg = ipywidgets.IntProgress(min=0, max=self.Nsteps, description='Running:',
        bar_style='', # 'success', 'info', 'warning', 'danger' or ''
        orientation='horizontal')
        display(wProg)
        
        TIME   = 0.
        INC    = 0
        outIND = -1
        Lp = self.L
        
        while INC < self.Nsteps:
            # update time
            TIME += self.dt
            # update step counter
            INC  += 1
            # update deformation gradient
            self.Fp *= np.eye(3) + Lp*self.dt
            # update the progress bar
            wProg.value += 1
            # is this step an output step?
            if INC in self.output_steps:
                output_step = True
                outIND += 1
            else:
                 output_step = False
                  
            self.crss_mean   = 0.
            
            for gID, grain in enumerate(self.grains):

               # rotate Lp by Q matrix from global to local (crystal) coord system
                Lp_loc = grain.Q @ Lp @ grain.Q.T
                # symmetric part of Lp - deformation rate tensor
                grain.Dp = getsym(Lp_loc)   
                grain.Dp_vec = m2voigt_dev(grain.Dp)
                # skewsymmetric part of Lp - total spin tensor
                Wp = getskw(Lp_loc)
        
                # run crystal plasticity for one grain (single crystal plasticity)
                grain.solveSingleCrystal(self.result_vars, solve_Tayloramb=self.options['solve_Tayloramb'], 
                                                           SRS=self.options['SRS'], 
                                                           gm0=self.options['gmdot0'])
                
                # update total slip for grain
                grain.total_slip += grain.total_sliprate*self.dt
                
                if (self.hardening_law['model'] != 'RIGID_PLASTIC') and (self.tmax != 0):
                    # update of the hardening state variables for grain
                    grain.hardening(self.dt)
                        
                # transform Cauchy stress to global coord sys
                grain.stress_glob = grain.Q.T @ grain.stress_loc @ grain.Q
                
                # calculate spin tensor from slip activity
                w = np.dot(crystal_properties.Omega, grain.sliprates)
                W_slip = np.array([[ 0.  ,  w[2], w[1]],
                                   [-w[2],  0.  , w[0]],
                                   [-w[1], -w[0], 0.  ]])

                W_lattice = Wp - W_slip

                # update grain orientation (2nd order scheme)
#                 grain.R = np.linalg.inv(np.eye(3) - W_lattice*self.dt/2.) @ (np.eye(3) + W_lattice*self.dt/2.) @ grain.R
#                 grain.Q = grain.R.T @ grain.Q0
                # update grain orientation (incremental scheme using matrix exponential)
                grain.R = scipy.linalg.expm(W_lattice*self.dt)
                grain.Q = grain.R.T @ grain.Q
                
                # calculate average crss for the whole polycrystal - used as a stress scale in convergence criterions
                self.crss_mean += np.mean(grain.crss)/self.Ngrains
                
                # saving the results
                if output_step:
                    if 'average_stress' in self.result_vars['polycrystal_results']:
                        self.average_stress[:,:,outIND] += grain.stress_glob/self.Ngrains
                    if 'average_slip' in self.result_vars['polycrystal_results']:
                        self.average_slip[outIND] += np.sum(grain.sliprates)*self.dt/self.Ngrains
                    grain.save_results(self.result_vars, outIND)
            
        
    
    
    def Alamel(self):
        wProg = ipywidgets.IntProgress(min=0, max=self.Nsteps, description='Running:',
        bar_style='', # 'success', 'info', 'warning', 'danger' or ''
        orientation='horizontal')
        display(wProg)
        
        TIME   = 0.
        INC    = 0
        outIND = -1
        Lp = self.L
        
        while INC < self.Nsteps:
            # update time
            TIME += self.dt
            # update step counter
            INC  += 1
            # update deformation gradient
            self.Fp = (np.eye(3) + Lp*self.dt) @ self.Fp
            # update the progress bar
            wProg.value += 1
            # is this step an output step?
            if INC in self.output_steps:
                output_step = True
                outIND += 1
            else:
                 output_step = False
                    
            self.crss_mean   = 0.
               
            for cID, cluster in enumerate(self.clusters):
                g1 = cluster.g1
                g2 = cluster.g2
                gb = cluster.gb
              
                # rotate Lp into local (crystal) coord system for each grain
                g1.Dp = getsym(g1.Q @ Lp @ g1.Q.T)
                g2.Dp = getsym(g2.Q @ Lp @ g2.Q.T)

                g1.Dp_vec = m2voigt_dev(g1.Dp)
                g2.Dp_vec = m2voigt_dev(g2.Dp)
                
                # rotate Lp from global to local coord system given by orientation of the grain boundary between the grain pairs) 
                Lp_gb = gb.Q @ Lp @ gb.Q.T
                
                # rotate Schmidt matrix from grain to grain boundary coord sys
                g1.Qb = gb.Q @ g1.Q.T
                g2.Qb = gb.Q @ g2.Q.T
                
                H = g1.Qb
                # rotated symR1 into grain1 coord sys - Voigt notation
                symR1g1 = np.array([ H[0,0]*H[2,0], H[0,1]*H[2,1], 0.5*(H[0,1]*H[2,2]+H[0,2]*H[2,1]), 0.5*(H[0,0]*H[2,2]+H[0,2]*H[2,0]), 0.5*(H[0,0]*H[2,1]+H[0,1]*H[2,0]) ])
                symR2g1 = np.array([ H[1,0]*H[2,0], H[1,1]*H[2,1], 0.5*(H[1,1]*H[2,2]+H[1,2]*H[2,1]), 0.5*(H[1,0]*H[2,2]+H[1,2]*H[2,0]), 0.5*(H[1,0]*H[2,1]+H[1,1]*H[2,0]) ])
                H = g2.Qb
                # rotated symR1 into grain1 coord sys - Voigt notation
                symR1g2 = np.array([ H[0,0]*H[2,0], H[0,1]*H[2,1], 0.5*(H[0,1]*H[2,2]+H[0,2]*H[2,1]), 0.5*(H[0,0]*H[2,2]+H[0,2]*H[2,0]), 0.5*(H[0,0]*H[2,1]+H[0,1]*H[2,0]) ])
                symR2g2 = np.array([ H[1,0]*H[2,0], H[1,1]*H[2,1], 0.5*(H[1,1]*H[2,2]+H[1,2]*H[2,1]), 0.5*(H[1,0]*H[2,2]+H[1,2]*H[2,0]), 0.5*(H[1,0]*H[2,1]+H[1,1]*H[2,0]) ])

                # find slips minimizing the total internal energy for the whole cluster including relaxations by simplex linear programming method
                cluster.findslips_relaxed(symR1g1, symR2g1, symR1g2, symR2g2)
                # caluclate stresses in the grains
                cluster.getstress(symR1g1, symR2g1, symR1g2, symR2g2)
                # calculate average crss for the whole polycrystal - a kind of stress scale
                self.crss_mean += (g1.crss + g2.crss)/self.Ngrains
                
                for i, g in enumerate((g1, g2)):
                    # velocity gradients after relaxation for each grain  (expressed in grain boundary sys)
                    Lp_gb_relaxed = Lp_gb - crystal_properties.R1*g.relaxation_sliprates[0] - crystal_properties.R2*g.relaxation_sliprates[1]
                    Lp_g_relaxed = g.Qb.T @ Lp_gb_relaxed @ g.Qb   # into grain coord sys 
                    g.Dp = getsym(Lp_g_relaxed)
                    g.Dp_vec = m2voigt_dev(g.Dp)
                    Wp = getskw(Lp_g_relaxed)

                    # solve Taylor ambiguity by singular value decomposition method
#                     g.solveambSVD()
                    
                    # update total slip for both grains in the cluster
                    g.total_slip += g.total_sliprate*self.dt
                    # update of the hardening state variables for both grains
                    if self.hardening_law['model'] != 'RIGID_PLASTIC':
                        g.hardening(self.dt)
                    
                    # calculate spin tensor from slip activity
                    w = np.dot(crystal_properties.Omega, g.sliprates)
                    W_slip = np.array([[ 0.  ,  w[2], w[1]],
                                       [-w[2],  0.  , w[0]],
                                       [-w[1], -w[0], 0.  ]])
                    W_lattice = Wp - W_slip
                    W_lattice = g.R @ W_lattice @ g.R.T
                    g.R = np.dot(np.linalg.inv(np.eye(3)-W_lattice*self.dt/2.),np.dot((np.eye(3)+W_lattice*self.dt/2.), g.R))
#                     g.R = scipy.linalg.expm(W_lattice*self.dt)   # give identical result as the second-order scheme above
                    g.Q = g.R.T @ g.Q0

                # rotate grain boundary
                if self.rotate_boundary:
                    gb.updateOrientation(self.Fp)
            
                # saving the results
                if output_step:
                    for i, g in enumerate((g1, g2)):
                        if 'average_stress' in self.result_vars['polycrystal_results']:
                            self.average_stress[:,:,outIND] += g.stress_glob/self.Ngrains
                        if 'average_slip' in self.result_vars['polycrystal_results']:
                            self.average_slip[outIND] += np.sum(g.sliprates)*self.dt/self.Ngrains
                        g.save_results(self.result_vars, outIND)  
        
    
    def read_input_orientations(self):

        if type(self.orientations) is dict: 
            ori = self.orientations['grains']
            if 'grain_boundaries' in self.orientations.keys():
                gb  = self.orientations['grain_boundaries']
                self.assign_orientations(ori, gb)
            else:
                self.assign_orientations(ori)
        else:
            self.assign_orientations(self.orientations)
            
            
    def assign_orientations(self, grain_ori, gb_ori=None):
        
        grain_ori = np.asarray(grain_ori)
        if len(grain_ori.shape) == 1:
            grain_ori = np.reshape(grain_ori,(1,len(grain_ori)))
        tmp_grains = []
        tmp_grain_boundaries = []
        for i, ori in enumerate(grain_ori):
            grain = Grain(ori, self.hardening_law)
            tmp_grains.append(grain)
        if gb_ori is not None:
            gb_ori = np.asarray(gb_ori)
            if len(gb_ori.shape) == 1:
                gb_ori = np.reshape(gb_ori,(1,len(gb_ori)))
            for i, ori in enumerate(gb_ori):
                gb = Grain_boundary(ori)
                tmp_grain_boundaries.append(gb)
        else:
            tmp_grain_boundaries = []
        
        if self.grain_interaction in ('ALAMEL','ALAMEL3'):
            if len(tmp_grains) % 2 != 0:
                tmp_grains.pop()
                print('You specified odd number of grains for ALAMEL-type two-grain interaction model. The last orientation will therefore be ommitted.')
            if  gb_ori is not None:
                diff = len(tmp_grain_boundaries) - len(tmp_grains)//2 
                if  diff < 0:
                    picked = np.random.randint(0, len(tmp_grain_boundaries), -diff)
                    for p in picked:
                        ori = tmp_grain_boundaries[p].init_euler_angles
                        gb = Grain_boundary(ori)
                        tmp_grain_boundaries.append(gb)
                elif diff > 0:
                    tmp_grain_boundaries = tmp_grain_boundaries[:-diff]
            else:     
                gb_ori = generate_random_orientations(len(tmp_grains)//2, 'angles')
                for i, ori in enumerate(gb_ori):
                    gb = Grain_boundary(ori)
                    tmp_grain_boundaries.append(gb)
                print('No grain boundary orientations specified for ALAMEL-type two-grain interaction model. Those will be assigned as random.')
                
            # create the clusters
            self.clusters = []
            for i in range(len(tmp_grain_boundaries)): 
                cl = Cluster(tmp_grains[2*i], tmp_grains[2*i+1], tmp_grain_boundaries[i])
                self.clusters.append(cl)
            self.Nclusters = len(self.clusters)
            self.Ngrains = 2*self.Nclusters
        else:
            self.grains = tmp_grains    
            self.Ngrains = len(self.grains)
    
    
    def rotate(self, R):
        if self.grain_interaction == 'FCTAYLOR':
            for g in self.grains:
                 g.Q = g.Q @ R
        else:
            for c in self.clusters:
                c.g1.Q = c.g1.Q @ R
                c.g2.Q = c.g2.Q @ R
#                 c.gb.Q = c.gb.Q @ R.T

                
   
    # calculate r-values
    def getRvalues(self, tensile_axis=[1,0,0], normal_axis=[0,0,1], 
                   options = {'increment_jacobian': 1.e-3, 
                              'rotate_boundary'    : True,
                              'use_SCYL'           : False,
                              'SCYL_exponent'      : 100,
                              'solve_Tayloramb'    : 'RD',
                              'gmdot0'             : 1.,
                              'SRS'                : 0.01}):
        R = np.zeros((3,3))
        tensile_axis = np.asarray(tensile_axis)/np.linalg.norm(tensile_axis)
        normal_axis = np.asarray(normal_axis)/np.linalg.norm(normal_axis)
        if not np.isclose(np.dot(tensile_axis,normal_axis), 0.):
            sys.exit('Tensile_axis and normal_axis are not perpendicular.')
        transv_axis = np.cross(normal_axis,tensile_axis)
        R[:,0] = tensile_axis
        R[:,1] = transv_axis
        R[:,2] = normal_axis
        if np.linalg.det(R) < 0:
            print('Determinant negative')
        self.rotate(R)
        
        sdir = [2./3., -1./3., 0., 0., 0.]
        self.init_cpfort()
        # boundary condictions
        cpfort.globals.ind_sdir = [True, True, False, False, False]
        cpfort.globals.ind_sabs = [False, False, True, True, True]
        cpfort.globals.ind_d    = [False, False, False, False, False]
        cpfort.globals.sdir_prescribed   = sdir
        cpfort.globals.sabs_prescribed   = np.zeros(5)
        cpfort.globals.d_prescribed      = np.zeros(5)
        cpfort.globals.w_prescribed      = np.zeros((3,3))
        cpfort.globals.eps4jacobian      = options['increment_jacobian']
        cpfort.scylglobals.scylon        = options['use_SCYL']
        cpfort.scylglobals.scylexp       = options['SCYL_exponent']
        cpfort.globals.grain_interaction = grain_interaction_dict[self.grain_interaction]
        Dsolved, Ssolved, info, resid, nfev = cpfort.crystal_plasticity.solve_mixbc(dguess=sdir, n=5)
        self.rotate(R.T)
        
        rvalue = Dsolved[1,1]/Dsolved[2,2]
        yield_stress = Ssolved[0,0]-(Ssolved[1,1]+Ssolved[2,2])/2
        
        return rvalue, yield_stress
    
    
        
        
    # generate yield locus
    def yield_locus(self, locus_type=None, user_input=None, input_type='strain', number_of_points=None, plot_axes=None, 
                    options = {'increment_jacobian': 1.e-3, 'tol': 1., 'use_SCYL':False, 'SCYL_exponent':100}):
        
        locus_type = locus_type.lower()
        self.init_cpfort()
        
        if locus_type == '2d':
            normal_components = [(0,0),(1,1),(2,2)]
            shear_components  = [(1,2),(0,2),(0,1)]
            
            xij = [int(x)-1 for x in plot_axes[0]]
            xij.sort()
            xij = tuple(xij)
            yij = [int(x)-1 for x in plot_axes[1]]
            yij.sort()
            yij = tuple(yij)
            xi, xj = xij
            yi, yj = yij
            if len(plot_axes) == 4:
                zij = [int(x)-1 for x in plot_axes[2]]
                zij.sort()
                zij = tuple(zij)
                zi, zj = zij
                if zij in [xij, yij]:
                    sys.exit('Out-of-plane stress component must differ.')
                elif zij not in normal_components + shear_components:
                    sys.exit('Out-of-plane stress component is NA.')
                s0_list = plot_axes[3]
            else:
                zij = None
                s0_list = [0.]

            wProg = ipywidgets.IntProgress(min=0, max=number_of_points*len(s0_list), description='Running:',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal')
            display(wProg)    
                
            angles = np.linspace(0, 2.*np.pi, number_of_points)
            YL = np.zeros((3,3,number_of_points*len(s0_list)))
            R = np.zeros(number_of_points*len(s0_list))
            non_converg_stress = 0
            
            for s0i, s0 in enumerate(s0_list):
                for ind, ang in enumerate(angles):
                    wProg.value += 1
#                     print('')
#                     print('{}: angle {}'.format(ind, np.rad2deg(ang)))
#                     print('')
                    Sdir = np.zeros((3,3))
                    Sabs = np.zeros((3,3))
                    ind_Sdir = np.full((3, 3), False, dtype=bool)
                    ind_Sabs = np.full((3, 3), True, dtype=bool)
                    
                    if xij in shear_components: # xij is a shear component
                        if yij in shear_components: # both are shear stress components
                            Sdir[xi, xj], Sdir[xj, xi] = np.cos(ang), np.cos(ang)
                            Sdir[yi, yj], Sdir[yj, yi] = np.sin(ang), np.sin(ang)
                            ind_Sdir[xi, xj], ind_Sdir[xj, xi] = True, True
                            ind_Sdir[yi, yj], ind_Sdir[yj, yi] = True, True
                            ind_Sabs[xi, xj], ind_Sabs[xj, xi] = False, False
                            ind_Sabs[yi, yj], ind_Sabs[yj, yi] = False, False
                            if zij in normal_components:
                                Sabs[zi,zj] = 2./3.*s0
                                rest_normal = [normal for normal in normal_components if normal != zij]
                                for rn in rest_normal:
                                    Sabs[rn[0], rn[1]] = -1./3.*s0
                            elif zij in shear_components:
                                Sabs[zi,zj] = s0
                                Sabs[zj,zi] = s0
                            elif zij is not None:
                                sys.exit('Yield surface plotting problem...')
                            
                        else: # yij is a normal component
                            Sdir[xi, xj] = np.cos(ang)
                            Sdir[xj, xi] = np.cos(ang)
                            Sdir[yi, yj] = 2./3.*np.sin(ang)
                            ind_Sdir[xi, xj], ind_Sdir[xj, xi] = True, True
                            ind_Sdir[yi, yj] = True
                            ind_Sabs[xi, xj], ind_Sabs[xj, xi] = False, False
                            ind_Sdir[yi, yj] = False
                            rest_normal = [normal for normal in normal_components if normal != yij]
                            for rn in rest_normal:
                                Sdir[rn[0], rn[1]] = -1./3.*np.sin(ang)
                                ind_Sdir[rn[0], rn[1]] = True
                                ind_Sabs[rn[0], rn[1]] = False
                            if zij in normal_components:
                                print('Out-of-plane stress component cannot be a normal component for this plot.')
                            elif zij in shear_components:
                                Sabs[zi, zj] = s0
                                Sabs[zj, zi] = s0
                            elif zij is not None:
                                sys.exit('Yield surface plotting problem...')
                                
                    else: # xij is a normal component
                        if yij in shear_components: # yij is a shear component
                            Sdir[yi, yj] = np.sin(ang)
                            Sdir[yj, yi] = np.sin(ang)
                            Sdir[xi, xj] = 2./3.*np.cos(ang)
                            ind_Sdir[yi, yj], ind_Sdir[yj, yi] = True, True
                            ind_Sdir[xi, xj] = True
                            ind_Sabs[yi, yj], ind_Sabs[yj, yi] = False, False
                            ind_Sabs[xi, xj] = False
                            rest_normal = [normal for normal in normal_components if normal != xij]
                            for rn in rest_normal:
                                Sdir[rn[0], rn[1]] = -1./3.*np.cos(ang)
                                ind_Sdir[rn[0], rn[1]] = True
                                ind_Sabs[rn[0], rn[1]] = False
                            if zij in normal_components:
                                print('Out-of-plane stress component cannot be a normal component for this plot.')
                            elif zij in shear_components:
                                Sabs[zi, zj] = s0
                                Sabs[zj, zi] = s0
                            elif zij is not None:
                                sys.exit('Yield surface plotting problem...')
                                
                        else: # both are normal stress components
                            rest_normal = [normal for normal in normal_components if normal not in [xij, yij]]
                            rest_normal = rest_normal[0] # extract the only tuple from the list
                            Sdir[xi, xj] = 2./3.*np.cos(ang) - 1./3.*np.sin(ang)
                            Sdir[yi, yj] = 2./3.*np.sin(ang) - 1./3.*np.cos(ang)
                            Sdir[rest_normal[0], rest_normal[1]] = -1./3.*np.cos(ang) - 1./3.*np.sin(ang)
                            ind_Sdir[0, 0], ind_Sdir[1, 1], ind_Sdir[2, 2] = True, True, True
                            ind_Sabs[0, 0], ind_Sabs[1, 1], ind_Sabs[2, 2] = False, False, False
                            if zij in normal_components:
                                print('Out-of-plane stress component cannot be a normal component for this plot.')
                            elif zij in shear_components:
                                Sabs[zi, zj] = s0
                                Sabs[zj, zi] = s0
                            elif zij is not None:
                                sys.exit('Yield surface plotting problem...')
                                

                    # boundary condictions
                    cpfort.globals.ind_sdir = m2voigt_dev(ind_Sdir)
                    cpfort.globals.ind_sabs = m2voigt_dev(ind_Sabs)
                    cpfort.globals.ind_d    = np.full(5, False, dtype=bool)
                    cpfort.globals.sdir_prescribed = m2voigt_dev(Sdir)
                    cpfort.globals.sabs_prescribed = m2voigt_dev(Sabs)
                    cpfort.globals.d_prescribed    = np.zeros(5)
                    cpfort.globals.w_prescribed    = np.zeros((3,3))
                    cpfort.globals.eps4jacobian = options['increment_jacobian']
                    cpfort.globals.grain_interaction = grain_interaction_dict[self.grain_interaction]
                    cpfort.scylglobals.scylon   = options['use_SCYL']
                    cpfort.scylglobals.scylexp  = options['SCYL_exponent']
                    
                    # initial guess for Dp
                    if ind == 0:
                        dguess = m2voigt_dev(Sdir)
                    else:
                        dguess = make_good_guess(Sabs, ind_Sabs, Sdir, ind_Sdir, cpfort.globals.crss_mean)
                        dguess = m2voigt_dev(dguess)
                        
                    Dsolved, Ssolved, info, resid, nfev = cpfort.crystal_plasticity.solve_mixbc(dguess=dguess, n=5)
                    print('Info: {}'.format(info))
                    print('Residuals: {}'.format(resid))
#                     print('Nfev: {}'.format(nfev))
                    
                    # after iteration process, save the found homogenized stress "Ssolved" to YL
                    R[ind+number_of_points*s0i] = max(abs(resid))
                    if max(abs(resid)) < options['tol']:
                        YL[:,:,ind+number_of_points*s0i] = Ssolved
                    else:
                        YL[:,:,ind+number_of_points*s0i] = np.full((3,3), np.nan)   
                        non_converg_stress += 1
            rij = normal_components.copy()
            if xij in rij:
                rij.remove(xij)
            if yij in rij:
                rij.remove(yij)
            if len(rij) == 1:
                rij = 2*rij
            if len(rij) == 3:
                rij = None
            if rij is not None:
                tmp = 0.5*(YL[rij[0][0],rij[0][1],:] + YL[rij[1][0],rij[1][1],:])
                YL[0,0,:] -= tmp
                YL[1,1,:] -= tmp
                YL[2,2,:] -= tmp

        elif locus_type == 'full':
            data = np.load('out5D.npy')
            number_of_points = min(number_of_points, data.shape[0])
            YL = np.zeros((3,3,number_of_points))
            R = np.zeros(number_of_points)
            np.random.randint(0,data.shape[0],number_of_points)
            random_choice = np.arange(data.shape[0])
            np.random.shuffle(random_choice)
            random_choice = random_choice[:number_of_points]
            user_input = data[random_choice,:]
            cpfort.globals.grain_interaction = grain_interaction_dict[self.grain_interaction]
            YL, NYLpoints = cpfort.crystal_plasticity.ylfull(user_input.T, input_type, number_of_points)
            YL = YL[:,:NYLpoints]
            YL = voigt2m_dev(YL)
            non_converg_stress = number_of_points - NYLpoints
        
        elif locus_type == 'user':
            number_of_points = user_input.shape[0]
            YL = np.zeros((3,3,number_of_points))
            R = np.zeros(number_of_points)
            cpfort.globals.grain_interaction = grain_interaction_dict[self.grain_interaction]
            YL, NYLpoints = cpfort.crystal_plasticity.ylfull(user_input.T, input_type, number_of_points)
            YL = YL[:,:NYLpoints]
            YL = voigt2m_dev(YL)
            non_converg_stress = number_of_points - NYLpoints
        else:
            sys.exit('locus_type not specified. Use "full", "user" or "2D".')
        
        return YL, R, non_converg_stress

        
    # texture visualization
    def plot_orientations(self, plot_type='IPF', marker='.', markersize='1', color='b', levelsODF=None):
        if plot_type.upper() == 'PF':
            # plot circle
            a1 = np.arange(0,2*np.pi,0.01)
            plt.plot(np.cos(a1), np.sin(a1),'k')
            
            if self.grain_interaction != 'FCTAYLOR':
                for cluster in self.clusters:
                    for grain in (cluster.g1, cluster.g2):
                        grain.plot_orientation('PF', plot_border=False, marker=marker, markersize=markersize, color=color)
            else:
                for grain in self.grains:
                    grain.plot_orientation('PF', plot_border=False, marker=marker, markersize=markersize, color=color)
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.figure(figsize=(40,40))
        
        elif plot_type.upper() == 'IPF':
            # plot individual orientations in Inverse Pole figure
            x3 = 1./np.sqrt(3.)/(1./np.sqrt(3.)+1.)
            x2 = 1./np.sqrt(2.)/(1./np.sqrt(2.)+1.)

            # plot the triangle
            plt.plot([0., x2], [0., 0.],'k')
            plt.plot([0., x3], [0., x3],'k')
            plt.axis([-0.05, 0.5, -0.05, 0.5])

            # plot arc
            a1 = np.arange(0., 0.263, 0.001)
            plt.plot((1.+x2)*np.cos(a1)-1.,(1.+x2)*np.sin(a1),'k')
            
            if self.grain_interaction != 'FCTAYLOR':
                for cluster in self.clusters:
                    for grain in (cluster.g1, cluster.g2):
                        grain.plot_orientation('IPF', plot_border=False, marker=marker, markersize=markersize, color=color)
            else:
                for grain in self.grains:
                    grain.plot_orientation('IPF', plot_border=False, marker=marker, markersize=markersize, color=color)
                
            plt.gca().set_aspect('equal', adjustable='box')
            
        elif plot_type.upper() == 'ODF':
            angs = np.zeros((int(self.Ngrains),3))
            if self.grain_interaction != 'FCTAYLOR':
                for i, c in enumerate(self.clusters):
                    angs[2*i,:] = matrix2ang(c.g1.Q)
                    angs[2*i+1,:] = matrix2ang(c.g2.Q)
            else:
                for i, g in enumerate(self.grains):
                    angs[i,:] = matrix2ang(g.Q)
            ori = odflib.Orientations(angles=angs)
            odf = odflib.ODF(orientations=ori)
            odf.show(boundaries=levelsODF)
    
      

    def elast_modulus(self):
    # caluclate 6x6 elastic tensor in voigt notation
        if self.elasticity is not None:
            if len(self.elasticity) == 3: # cubic symmetry
                c11, c12, c44 = self.elasticity
                self.C  = np.array( [[ c11, c12, c12,  0.,  0.,  0.],
                                     [ c12, c11, c12,  0.,  0.,  0.],
                                     [ c12, c12, c11,  0.,  0.,  0.],
                                     [  0.,  0.,  0., c44,  0.,  0.],
                                     [  0.,  0.,  0.,  0., c44,  0.],
                                     [  0.,  0.,  0.,  0.,  0., c44]])
            elif len(self.elasticity) == 2: # isotropy
                E, mu = self.elasticity
                lamb = E*mu/((1+mu)*(1-2.*mu))
                c11 = 2.*mu + lamb
                c12 = lamb
                c44 = mu
                self.C  = np.array( [[ c11, c12, c12,  0.,  0.,  0.],
                                     [ c12, c11, c12,  0.,  0.,  0.],
                                     [ c12, c12, c11,  0.,  0.,  0.],
                                     [  0.,  0.,  0., c44,  0.,  0.],
                                     [  0.,  0.,  0.,  0., c44,  0.],
                                     [  0.,  0.,  0.,  0.,  0., c44]])
            else:
                print('Elasticity model not recognized. Purely plastic model will be used.')
                self.elasticity = None
                self.C = None
        else:
            self.C = None                   
    
    
    # --------- END OF THE CLASS METHOD DEFINITION --------- #
    
    


    
def anyis(x, value, tol):
    return np.any(np.abs(x - value) <= value*tol)

def alleq(x, value, tol):
    return np.all(np.abs(x - value) <= value*tol)

def allless(x, value, tol):
    return np.all(np.abs(x) <= value*(1.-tol))


def elast_stressinc(C, De):
    # multiply elastic strain rate with elastic tensor
    # and convert stress rate into matrix notation
    return voigt2m(np.dot(C, m2voigt(De, tensor='strain')), tensor='stress')
    

def ori2IPF(Q, direction, points):
    eps = 1.e-12
    x2, x3 = points
    vec = np.dot(Q, direction)
    permuts = itertools.permutations(vec)
    for p in permuts:
        p = np.array(p)
        # to have it on northern hemisphere on the correct quadrant
        p = p*np.sign(np.array([p[0], -p[1], p[2]]))   
        proj111x = p[0]/(p[2]+1.)
        proj111y = p[1]/(p[2]+1.)
        xcoord = -proj111y      
        ycoord =  proj111x
        if (xcoord <= x2+eps) and (ycoord <= xcoord+eps) and \
           (np.sqrt((1.+xcoord)**2 + ycoord**2) <= (1.+x2)+eps):
            return xcoord, ycoord
            break
    
def make_good_guess(Sabs, iSabs, Sdir, iSdir, crssmean):
    Sdir = np.array(Sdir,copy=True)
    Sabs = np.array(Sabs,copy=True)
    
    if np.any(iSdir == True) or np.any(iSabs == True):
        vm = 3.*crssmean
        vm = 2./3.*(vm**2)
        tmp = 0.

        vm -= np.sum((Sabs[iSabs])**2)
        tmp += np.sum((Sdir[iSdir])**2)
        if (tmp > 0.):
            if (vm > 0.):
                k = np.sqrt(vm/tmp)
                mx = np.ma.masked_array(Sdir, mask=np.logical_not(iSdir))
                mx *= k
                S1 = np.ma.filled(mx, 0.)
                guess = S1 + Sabs
                guess = VonMises(guess, meassure='strain', normalize=True)
            else: # VonMises norm of all Sabs elements is more than 3*CRSS
                guess = Sdir + Sabs
                guess = VonMises(guess, meassure='strain', normalize=True)
        else: # there is no Sdir
            guess = Sabs
            guess = VonMises(guess, meassure='strain', normalize=True)
    else:
        guess = np.zeros((3,3))

    return guess


def ang2matrix(angles_in_degrees):
    # input angles in deg
    phi1, PHI, phi2 = np.deg2rad(angles_in_degrees)
    # partial rotation matrices
    C = np.array([[ np.cos(phi1),  np.sin(phi1),    0.0],
                  [-np.sin(phi1),  np.cos(phi1),    0.0],
                  [      0.0    ,      0.0     ,    1.0]])
         
    B = np.array([[      1.0    ,      0.0     ,    0.0],
                  [      0.0    ,  np.cos(PHI) , np.sin(PHI)],
                  [      0.0    , -np.sin(PHI) , np.cos(PHI)]])
     
    A = np.array([[ np.cos(phi2),  np.sin(phi2),    0.0],
                  [-np.sin(phi2),  np.cos(phi2),    0.0],
                  [      0.0    ,      0.0     ,    1.0]])
    # rotation matrix
    Q = np.dot(A,np.dot(B,C))
    return Q


def matrix2ang(Q):
    if np.abs(Q[2,2]) < 1.:
        ANG2 = np.arccos(Q[2,2])
        STH  = np.sin(ANG2)
        ANG1 = np.arctan2(Q[2,0]/STH, -Q[2,1]/STH)
        ANG3 = np.arctan2(Q[0,2]/STH, Q[1,2]/STH)
    else:
        ANG1 = np.arctan2(Q[0,1],Q[0,0])
        ANG2 = 0.
        ANG3 = 0.
    
    # output angles in deg
    return np.rad2deg([ANG1, ANG2, ANG3])


def axis_angle2matrix(axis, ang):
    axis = np.asarray(axis)
    r1, r2, r3 = axis/np.linalg.norm(axis)
    ang = np.deg2rad(ang)
    return np.array([[(1.-r1**2)*np.cos(ang)+r1**2,          r1*r2*(1.-np.cos(ang))+r3*np.sin(ang), r1*r3*(1.-np.cos(ang))-r2*np.sin(ang)],
                     [r1*r2*(1.-np.cos(ang))-r3*np.sin(ang), (1.-r2**2)*np.cos(ang)+r2**2,          r2*r3*(1.-np.cos(ang))+r1*np.sin(ang)],
                     [r1*r3*(1.-np.cos(ang))+r2*np.sin(ang), r2*r3*(1.-np.cos(ang))-r1*np.sin(ang), (1.-r3**2)*np.cos(ang)+r3**2]])

def spherical_spread(euler_angles, spread=15, N=10000, ortho=True):
    Q0 = ang2matrix(euler_angles)
    if ortho: 
        N = N//4
        O1 = np.diag([1,-1,-1])
        O2 = np.diag([-1,1,-1])
        O3 = np.diag([-1,-1,1])
    i = np.arange(0, N, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/N)
    goldenRatio = (1 + 5**0.5)/2
    theta = 2*np.pi * i / goldenRatio
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    n = np.empty((N,3))
    n[:,0] = x
    n[:,1] = y
    n[:,2] = z
    omg = np.deg2rad(spread)*np.random.randn(N)
    if ortho:
        oridata = np.empty((4*N,3))
    else:
        oridata = np.empty((N,3))

    for i in range(N):
        R = np.tensordot(n[i,:], n[i,:], axes=0)
        W = np.array([[0, -n[i,2], n[i,1]],
                     [n[i,2],  0, -n[i,0]],
                     [-n[i,1], n[i,0], 0]])
        R = R + (np.eye(3)-R)*np.cos(omg[i]) + W*np.sin(omg[i])
        R = R@Q0
        if ortho:
            oridata[4*i,:] = matrix2ang(R)
            oridata[4*i+1,:] = matrix2ang(O1@R@O1.T)
            oridata[4*i+2,:] = matrix2ang(O2@R@O2.T)
            oridata[4*i+3,:] = matrix2ang(O3@R@O3.T)
        else:
            oridata[i,:] = matrix2ang(R)
    return oridata


def fibre_spread(euler_angles, axis=[0,0,1], spread=15, Naxi=36, Nspread=50):
    spheredata = spherical_spread(euler_angles, spread, Nspread, ortho=False)
    oridata = np.empty((Naxi*Nspread,3))
    phis = np.linspace(0,360,Naxi)
    # phis = 360.*np.random.rand(Naxi)
    phis = phis[1:]
    oridata[:Nspread,:] = spheredata
    i = 0
    for phi in phis:
        R0 = axis_angle2matrix(axis, phi)
        for ori in spheredata:
            R = ang2matrix(ori)
            oridata[Nspread+i,:] = matrix2ang(R@R0)
            i += 1
    return oridata


def hkluvw2ang(hkl,uvw):
    hkl = np.asarray(hkl)/np.linalg.norm(hkl)
    uvw = np.asarray(uvw)/np.linalg.norm(uvw)
    dir3 = np.cross(hkl,uvw)
    Q = np.empty((3,3))
    Q[:,0] = uvw
    Q[:,1] = dir3
    Q[:,2] = hkl
    return matrix2ang(Q)

def VonMises(A, meassure='strain', normalize=False):
# von Mises norm of stress or strain
    A = np.asarray(A)
    if A.shape == (3,3):
        A = deviator(A)
        x = m2voigt(A)
    elif len(A) == 5:
        x = np.insert(A,2,-A[0]-A[1])
    elif len(A) == 6:
        x = A.copy()
        x[:3] = x[:3] - np.sum(x[:3]/3.)
    if meassure == 'strain':
        fVM = np.sqrt(2./3.)
    if meassure == 'stress':
        fVM = np.sqrt(3./2.)
    VM = fVM*np.sqrt(np.sum(x[:3]**2) + 2.*(np.sum(x[-3:]**2)))
    if normalize:
        return A/VM
    else:
        return VM      

def deviator(A):
    return A - np.sum(np.diag(A))*np.eye(3)/3.

def getsym(A):
    return 0.5*(A+A.T)

def getskw(A):
    return 0.5*(A-A.T)

def voigt2m(x):
    x = np.asarray(x)
    S = np.zeros((3,3))
    S[0,0] = x[0]
    S[1,1] = x[1]
    S[2,2] = x[2]    
    S[1,2] = x[3]
    S[0,2] = x[4]
    S[0,1] = x[5]    
    S[1,0] = S[0,1]
    S[2,0] = S[0,2]
    S[2,1] = S[1,2]    
    return S

def voigt2m_dev(x):
    x = np.asarray(x)
    xshp = x.shape
    if len(xshp) == 1:
        S = np.zeros((3,3))
        S[0,0] = x[0]
        S[1,1] = x[1]
        S[1,2] = x[2]
        S[2,1] = x[2]
        S[0,2] = x[3]
        S[2,0] = x[3]
        S[0,1] = x[4]
        S[1,0] = x[4]
        S[2,2] = -x[0]-x[1]
    elif len(xshp) == 2:
        S = np.zeros((3,3,xshp[1]))
        S[0,0,:] = x[0,:]
        S[1,1,:] = x[1,:]
        S[1,2,:] = x[2,:]
        S[2,1,:] = x[2,:]
        S[0,2,:] = x[3,:]
        S[2,0,:] = x[3,:]
        S[0,1,:] = x[4,:]
        S[1,0,:] = x[4,:]
        S[2,2,:] = -x[0,:]-x[1,:]
    return S

def m2voigt(M):
    M = np.asarray(M)
    return np.array([M[0,0], M[1,1], M[2,2], M[1,2], M[0,2], M[0,1]])

def m2voigt_dev(D):
    D = np.asarray(D)
    return np.array([D[0,0], D[1,1], D[1,2], D[0,2], D[0,1]])

def mandel2m(x):
    x = np.asarray(x)
    S = np.zeros((3,3))
    S[0,0] = x[0]
    S[1,1] = x[1]
    S[2,2] = x[2]    
    S[1,2] = x[3]/np.sqrt(2.)
    S[0,2] = x[4]/np.sqrt(2.)
    S[0,1] = x[5]/np.sqrt(2.)
    S[1,0] = x[5]/np.sqrt(2.)
    S[2,0] = x[4]/np.sqrt(2.)
    S[2,1] = x[3]/np.sqrt(2.)  
    return S

def m2mandel(M):
    M = np.asarray(M)
    return np.array([M[0,0], M[1,1], M[2,2], np.sqrt(2.)*M[1,2], np.sqrt(2.)*M[0,2], np.sqrt(2.)*M[0,1]])

def lequeu2m(x):
    x = np.asarray(x)
    S = np.zeros((3,3))
    S[0,0] = -1./np.sqrt(6.)*(np.sqrt(3.)*x[0]+x[1])
    S[1,1] =  1./np.sqrt(6.)*(np.sqrt(3.)*x[0]-x[1])
    S[1,2] = x[2]/np.sqrt(2.)
    S[2,1] = x[2]/np.sqrt(2.)
    S[0,2] = x[3]/np.sqrt(2.)
    S[2,0] = x[3]/np.sqrt(2.)
    S[0,1] = x[4]/np.sqrt(2.)
    S[1,0] = x[4]/np.sqrt(2.)
    S[2,2] = x[1]*np.sqrt(2./3.)
    return S

def m2lequeu(D):
    D = np.asarray(D)
    return np.array([(D[1,1]-D[0,0])/np.sqrt(2), np.sqrt(3./2.)*D[2,2], np.sqrt(2.)*D[1,2], np.sqrt(2.)*D[0,2], np.sqrt(2.)*D[0,1]])



    
def getstress2(activesID, crss):

    B = np.zeros((len(activesID),5))

    for i, ind in enumerate(activesID):
        tempP = crystal_properties.P3d[:,:,ind]
        B[i,:] = [2.*tempP[0,0]+tempP[1,1], 2.*tempP[1,1]+tempP[0,0], 
                  2.*tempP[1,2],
                  2.*tempP[0,2], 
                  2.*tempP[0,1]]

    x = np.linalg.solve(B, crss[activesID])
    S = voigt2m_dev(x)
    return S
    
def getstress(activesID, crss):
    '''
    Given 5 active slip systems "activesID" and the critical resolved
    shear stresses the function finds a projection matrix B in the pre-generated 
    lookup table "Scalc_lookup" and by solving a linear system of equations 
    (eq. 69 in https://doi.org/10.1016/j.ijplas.2013.10.002) it returns
    the stress tensor.
    '''
    row = crystal_properties.sslookup[tuple(activesID)]
    B = crystal_properties.Scalc_lookup[row,:].reshape(5,5)
    x = np.linalg.solve(B, crss[activesID])
    S = voigt2m_dev(x)
    return S

def getRSS(S, P):   
    rss = np.zeros(24)
    for i in range(24):
        rss[i] = np.tensordot(S, P[:,:,i], axes=2)
#     x = m2voigt_dev(S)
#     rss = np.dot(crystal_properties.S_proj, x) # both positive and negative
    return rss
    
def stress_average(stress_list):
    if type(stress_list) is list:
        Savg = np.average(np.asarray(stress_list),axis=0)
    else: 
        Savg = stress_list
    return Savg


def solveamb_QP(gm_list, *args):
    ''' 
    Function solveamb_QP solves the Taylor ambiguity by minimizing L2 norm of 
    slip solutions. 
    ------
    Inputs
    ------
    gm_list : array, shape=(N, k)
        Matrix of k basis solutions. N is number of slipsystems.
    -------    
    Returns
    -------
    gm_QP : array, shape=(N,)
        Solution to the QP.
    '''
    
    """
    solve_qp(double[:, :] G, double[:] a, double[:, :] C=None, double[:] b=None, int meq=0, factorized=False)
    
    Solve a strictly convex quadratic program
    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b
    This routine uses the the Goldfarb/Idnani dual algorithm [1].
    References
    ---------
    ... [1] D. Goldfarb and A. Idnani (1983). A numerically stable dual
        method for solving strictly convex quadratic programs.
        Mathematical Programming, 27, 1-33.
    Parameters
    ----------
    G : array, shape=(n, n)
        matrix appearing in the quadratic function to be minimized
    a : array, shape=(n,)
        vector appearing in the quadratic function to be minimized
    C : array, shape=(n, m)
        matrix defining the constraints under which we want to minimize the
        quadratic function
    b : array, shape=(m), default=None
        vector defining the constraints
    meq : int, default=0
        the first meq constraints are treated as equality constraints,
        all further as inequality constraints (defaults to 0).
    factorized : bool, default=False
        If True, then we are passing :math:`R^{−1}` (where :math:`G = R^T R`)
        instead of the matrix G in the argument G.
    Returns
    -------
    x : array, shape=(n,)
        vector containing the solution of the quadratic programming problem.
    f : float
        the value of the quadratic function at the solution.
    xu : array, shape=(n,)
        vector containing the unconstrained minimizer of the quadratic function
    iterations : tuple
        2-tuple. the first component contains the number of iterations the
        algorithm needed, the second indicates how often constraints became
        inactive after becoming active first.
    lagrangian : array, shape=(m,)
        vector with the Lagragian at the solution.
    iact : array
        vector with the indices of the active constraints at the solution.
    """

    ncols = gm_list.shape[1]
    reduction = False
    # remove the linear dependent basis solutions if there is more than 4 of them
    if ncols > 4:
        reduction = True
        w = np.zeros(ncols)
        gm_list, columns = LIreduction(gm_list)
        ncols = len(columns)
        
    G = 2.0*np.dot(gm_list.T, gm_list)

    # linear term is included via optional arguments
    if args:
        a = args[0]
    else:
        a = np.zeros((ncols,))
    
    Aeq = np.ones((1,ncols))
    beq = 1.

    Gineq = gm_list
    hineq = np.zeros((12,))
    
    C = np.vstack([Aeq, Gineq]).T
    b = np.hstack([beq, hineq])
    meq = Aeq.shape[0]

    x = quadprog.solve_qp(G, a, C, b, meq)[0]

    gm_QP = np.dot(gm_list,x)
    
    if reduction:
        w[columns] = x
    else:
        w = x
    
    return gm_QP, w


def LIreduction(m):
    ncols = m.shape[1]
    indpcol = [x for x in range(ncols)]
    while ncols > 4:
        for trycol in indpcol:
            cols = [x for x in indpcol if x!=trycol]
            if np.linalg.matrix_rank(m[:,cols]) == 4:
                indpcol.remove(trycol)
                ncols -= 1
                break
    return m[:,indpcol], indpcol






# nonlinear solvers of Taylor mabiguity #
def solveamb_m_norm(gm_list, expm):
    def func(w):
        gm = np.dot(gm_list,w)
        return np.sum((np.abs(gm))**expm)
    
    no_var = gm_list.shape[1]         
    
    x0 = (1./no_var)*np.ones(no_var)
    
    # Uses a Nelder-Mead simplex algorithm to find the minimum of function
#    # unconstrained
#    method = 'Nelder-Mead'
#    method = 'Powell'
#    # constrained
#    method = 'L-BFGS-B'
#    method = 'TNC'
#    method = 'COBYLA'
    method = 'SLSQP'
    
    bnds = tuple([(0.,1.)]*no_var)
    Aeq = np.ones(no_var)
    beq = 1.
    cons = {'type': 'eq',   'fun': lambda x:  np.dot(Aeq,x)-beq}
    sol = scipy.optimize.minimize(func, x0, args=(), method=method, jac=None, hess=None, hessp=None, bounds=bnds, constraints=cons, tol=1e-8, callback=None, options=None)
    if not sol.status == 0:
        print('RI solution (minimizing m-norm) problem!')
    
    return np.dot(gm_list,sol.x), sol.x



def getL(mode, refDir, TD, angle, r, epsdot):
    '''
    *****************************************************************************************************
    This function return a prescribed velocity gradient tensor with skew part assumed to be zero.

    INPUTS:
    ---------
    mode           - can be 'tensile' or 'rolling'
    refDir         - is reference direction, in case of tension it is a tension direction, 
                     in case of rolling it is a rolling direction
    TD             - means transverse direction for rolling mode, has no meaning for tensile mode 
    angle          - applies only for mode='tensile', rotation angle in refDir-TD plane 
                     counted from refDir towards TD
    rvalue         - is Lankford's r-value = D22/D33
    epsdot         - is prescribed Von Mises strain rate 
    
    OUTPUT:
    ---------
    L              - prescribed symmetric velocity gradient tensor in global CS
    *****************************************************************************************************
    '''
    angle = np.deg2rad(angle)
    
    if mode == 'tensile':
        if refDir=='x' : 
            L = np.array([[1.+(1.+2.*r)*np.cos(2.*angle),  (1.+2.*r)*np.sin(2.*angle),     0.],
                          [(1.+2.*r)*np.sin(2.*angle),     1.-(1.+2.*r)*np.cos(2.*angle),  0.],
                          [0.,                             0.,                            -2.]])*np.sqrt(3.)*epsdot/(4.*np.sqrt(r**2+r+1.))
        if refDir=='y':
            L = np.array([[-2,  0.,                             0.                           ],
                          [ 0., 1.+(1.+2.*r)*np.cos(2.*angle),  (1.+2.*r)*np.sin(2.*angle)   ],
                          [ 0., (1.+2.*r)*np.sin(2.*angle),     1.-(1.+2.*r)*np.cos(2.*angle)]])*np.sqrt(3.)*epsdot/(4.*np.sqrt(r**2+r+1.))
        if refDir=='z': 
            L = np.array([[1.-(1.+2.*r)*np.cos(2.*angle),  0.,  (1.+2.*r)*np.sin(2.*angle)   ],
                          [0.,                            -2.,  0.                           ],
                          [(1.+2.*r)*np.sin(2.*angle),     0.,  1.+(1.+2.*r)*np.cos(2.*angle)]])*np.sqrt(3.)*epsdot/(4.*np.sqrt(r**2+r+1.))
            
    if mode == 'rolling':
        if refDir=='x' and TD=='y': L = np.array([[ 1.0,  0.0,  0.0],[0.0,  0.0,  0.0],[0.0,   0.0, -1.0]])
        if refDir=='x' and TD=='z': L = np.array([[ 1.0,  0.0,  0.0],[0.0, -1.0,  0.0],[0.0,   0.0,  0.0]])
        if refDir=='y' and TD=='x': L = np.array([[ 0.0,  0.0,  0.0],[0.0,  1.0,  0.0],[0.0,   0.0, -1.0]])
        if refDir=='y' and TD=='z': L = np.array([[-1.0,  0.0,  0.0],[0.0,  1.0,  0.0],[0.0,   0.0,  0.0]])
        if refDir=='z' and TD=='x': L = np.array([[ 0.0,  0.0,  0.0],[0.0, -1.0,  0.0],[0.0,   0.0,  1.0]])
        if refDir=='z' and TD=='y': L = np.array([[-1.0,  0.0,  0.0],[0.0,  0.0,  0.0],[0.0,   0.0,  1.0]]) 
        
    return L
               

def generate_random_orientations(N, output='angles'):  # output 'angles' or 'matrix'
    angles = np.random.rand(N,3)

    angles[:,0] = 2.*np.pi*angles[:,0]
    angles[:,1] = np.arccos(2.*angles[:,1]-1.)
    angles[:,2] = 2.*np.pi*angles[:,2]
    
    if output == 'matrix':
        Q0 = np.zeros((N,3,3))
        for i in range(N):
            Q0[i,:,:] = ang2matrix(np.rad2deg(angles[i,:]))     
        return Q0 
    else: 
        return np.rad2deg(angles)
    

def getRotationForP(Q):
    A = np.array([[Q[0,0]**2-Q[0,2]**2, Q[0,1]**2-Q[0,2]**2, 2*Q[0,1]*Q[0,2], 2*Q[0,0]*Q[0,2], 2*Q[0,0]*Q[0,1]],
                  [Q[1,0]**2-Q[1,2]**2, Q[1,1]**2-Q[1,2]**2, 2*Q[1,1]*Q[1,2], 2*Q[1,0]*Q[1,2], 2*Q[1,0]*Q[1,1]],
                  [Q[1,0]*Q[2,0]-Q[1,2]*Q[2,2], Q[1,1]*Q[2,1]-Q[1,2]*Q[2,2], Q[1,1]*Q[2,2]+Q[1,2]*Q[2,1], Q[1,0]*Q[2,2]+Q[1,2]*Q[2,0], Q[1,0]*Q[2,1]+Q[1,1]*Q[2,0]],
                  [Q[0,0]*Q[2,0]-Q[0,2]*Q[2,2], Q[0,1]*Q[2,1]-Q[0,2]*Q[2,2], Q[0,1]*Q[2,2]+Q[0,2]*Q[2,1], Q[0,0]*Q[2,2]+Q[0,2]*Q[2,0], Q[0,0]*Q[2,1]+Q[0,1]*Q[2,0]],
                  [Q[0,0]*Q[1,0]-Q[0,2]*Q[1,2], Q[0,1]*Q[1,1]-Q[0,2]*Q[1,2], Q[0,1]*Q[1,2]+Q[0,2]*Q[1,1], Q[0,0]*Q[1,2]+Q[0,2]*Q[1,0], Q[0,0]*Q[1,1]+Q[0,1]*Q[1,0]]])
    return A
