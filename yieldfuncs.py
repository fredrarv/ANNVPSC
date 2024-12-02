import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, minimize, basinhopping, differential_evolution, shgo
from copy import copy, deepcopy
import multiprocessing as mp
import time
import sys
import crystal_plasticity_module as cp
import f2py_yldfit as fyf


def rotM(phi1, PHI, phi2):

    #rotation matrices
    C = np.array([[ np.cos(phi1),  np.sin(phi1),    0.0],
                  [-np.sin(phi1),  np.cos(phi1),    0.0],
                  [      0.0    ,      0.0     ,    1.0]])
         
    B = np.array([[      1.0    ,      0.0     ,    0.0],
                  [      0.0    ,  np.cos(PHI) , np.sin(PHI)],
                  [      0.0    , -np.sin(PHI) , np.cos(PHI)]])
     
    A = np.array([[ np.cos(phi2),  np.sin(phi2),    0.0],
                  [-np.sin(phi2),  np.cos(phi2),    0.0],
                  [      0.0    ,      0.0     ,    1.0]])
        
    # Euler matrix
    return np.dot(A,np.dot(B,C))

def fromnatural(s):
    '''
    From 5-vector representing a deviator into a 3x3 matrix representation
    '''
    return np.array([[1./np.sqrt(6.)*(-s[0]-np.sqrt(3.)*s[1]), s[4]/np.sqrt(2.), s[3]/np.sqrt(2.)],
                     [s[4]/np.sqrt(2.), 1./np.sqrt(6.)*(-s[0]+np.sqrt(3.)*s[1]), s[2]/np.sqrt(2.)],
                     [s[3]/np.sqrt(2.), s[2]/np.sqrt(2.), 1./np.sqrt(6.)*(2.*s[0])]])
def tonatural(s):
    '''
    From a 6-vector Voigt representation into a 5-vector natural deviatoric representation 
    '''
    return np.array([1./np.sqrt(6.)*(2.*s[2]-s[1]-s[0]), 
                     1./np.sqrt(2.)*(s[1]-s[0]), 
                     np.sqrt(2.)*s[3], 
                     np.sqrt(2.)*s[4], 
                     np.sqrt(2.)*s[5]])

def Voigt2matrix(s):
    return np.array([[s[0], s[5], s[4]],
                     [s[5], s[1], s[3]],
                     [s[4], s[3], s[2]]])

def YF(k, SIG, yfun, keep=None, dofort=True):
    if yfun['func'] == 'AXIFACET':
        model  = yfun['func']
        coef   = yfun['coef']
        m      = yfun['exp']
        Ncones = yfun['Ncones']
    elif yfun['func'] == 'FACET':
        model   = yfun['func']
        facets  = yfun['params']['facets']
        Nfacets = yfun['params']['Nfacets']
        exps    = yfun['params']['exps']
        N       = yfun['params']['N']
        q       = yfun['params']['q']
        s0s     = yfun['params']['s0s']
        pref    = yfun['params']['pref']
    elif yfun['func'] == 'FACETR':
        model    = yfun['func']
        facets0  = yfun['params']['facets0']
        Nfacets0 = yfun['params']['Nfacets0']
        exps0    = yfun['params']['exps0']
        N0       = yfun['params']['N0']
        s0s0     = yfun['params']['s0s0']
        pref     = yfun['params']['pref']
        q        = yfun['params']['q']
        facets   = yfun['params']['facets']
        exps     = yfun['params']['exps']
        N        = yfun['params']['N']
        s0s      = yfun['params']['s0s']
        ortho    = yfun['params']['ortho']
    else:
        model  = yfun['func']
        coef   = yfun['coef']
        m      = yfun['exp']
        NT     = yfun['Ntransf']
        y0     = yfun['y0']
    
    sig = k * np.array([SIG[0,0], SIG[1,1], SIG[2,2], SIG[1,2], SIG[2,0], SIG[0,1]])
    if keep is not None:
        Voigt = {'11':0, '22':1, '33':2, '23':3, '32':3, '13':4, '31':4, '12':5, '21':5}
        for comp in keep:
            zij = [int(x)-1 for x in comp]
            zij.sort()
            zi, zj = tuple(zij)
            sig[Voigt[comp]] = SIG[zi,zj]
    
    if dofort:
        if 'FACET' == model:
            f = fyf.facet(sig, facets, Nfacets, exps, N, s0s, pref, q)
            return f - 1.
        elif 'FACETR' == model:
            f = fyf.facetr(sig, facets0, Nfacets0, exps0, N0, s0s0, pref, q, facets, exps, s0s, N, ortho)
            return f - 1.
        
        elif 'YLD2004_09p' == model:
            if len(coef) == 3:
                # axisymmetric case
                coef_mod = np.zeros(9)
                # C12, C13, C44
                # C31 = 1
                coef_mod[[0,1,6]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[4] = 1.
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                f = fyf.yld2004_9p(sig, coef_mod, m)
            elif len(coef) == 9:
                f = fyf.yld2004_9p(sig, coef, m)
        elif 'YLD2004_18p' == model:
            if len(coef) == 7:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44
                coef_mod = np.zeros(18)
                coef_mod[[0,1,4,6,9,10,15]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                f = fyf.yld2004_18p(sig, coef_mod, m)
            elif len(coef) == 18:
                f = fyf.yld2004_18p(sig, coef, m)      
        elif 'YLD2011_18p' == model:
            if len(coef) == 7:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44
                coef_mod = np.zeros(18)
                coef_mod[[0,1,4,6,9,10,15]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                f = fyf.yld2011_18p(sig, coef_mod, m)
            elif len(coef) == 18:
                f = fyf.yld2011_18p(sig, coef, m)
        elif 'YLD2004_27p' in model:
            if len(coef) == 10:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44
                # D31 = 1
                # E31 = 1
                coef_mod = np.zeros(27)
                coef_mod[[0,1,4,6,9,10,15,18,19,24]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                coef_mod[20] = coef_mod[18]
                coef_mod[21] = coef_mod[19]
                coef_mod[22] = 1.
                coef_mod[23] = coef_mod[22]
                coef_mod[25] = coef_mod[24]
                coef_mod[26] = coef_mod[18]
                f = fyf.yld2004_27p(sig, coef_mod, m)
            elif len(coef) == 27:
                f = fyf.yld2004_27p(sig, coef, m)
        elif 'YLD2013_27p' in model:
            if len(coef) == 10:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44
                # D31 = 1
                # E31 = 1
                coef_mod = np.zeros(27)
                coef_mod[[0,1,4,6,9,10,15,18,19,24]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                coef_mod[20] = coef_mod[18]
                coef_mod[21] = coef_mod[19]
                coef_mod[22] = 1.
                coef_mod[23] = coef_mod[22]
                coef_mod[25] = coef_mod[24]
                coef_mod[26] = coef_mod[18]
                f = fyf.yld2013_27p(sig, coef_mod, m)
            elif len(coef) == 27:
                f = fyf.yld2013_27p(sig, coef, m)
        elif 'YLD2011_27p' == model:
            if len(coef) == 11:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44, E12, E13, E31, E44
                # D31 = 1
                coef_mod = np.zeros(27)
                coef_mod[[0,1,4,6,9,10,15,18,19,22,24]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                coef_mod[20] = coef_mod[18]
                coef_mod[21] = coef_mod[19]
                coef_mod[23] = coef_mod[22]
                coef_mod[25] = coef_mod[24]
                coef_mod[26] = coef_mod[18]
                f = fyf.yld2011_27p(sig, coef_mod, m)
            elif len(coef) == 27:
                f = fyf.yld2011_27p(sig, coef, m)
        elif model == 'YLD2013_36p_axi':
            if len(coef) == 13:
                # axisymmetric case
                # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44, F12, F13, F44
                # D31 = 1
                # E31 = 1
                # F31 = 1
                NT = 4
                coef_mod = np.zeros(200)
                coef_mod[[0,1,4,6,9,10,15,18,19,24,27,28,33]] = coef
                coef_mod[2] = coef_mod[0]
                coef_mod[3] = coef_mod[1]
                coef_mod[5] = coef_mod[4]
                coef_mod[7] = coef_mod[6]
                coef_mod[8] = coef_mod[0]
                #
                coef_mod[11] = coef_mod[9]
                coef_mod[12] = coef_mod[10]
                coef_mod[13] = 1.
                coef_mod[14] = coef_mod[13]
                coef_mod[16] = coef_mod[15]
                coef_mod[17] = coef_mod[9]
                #
                coef_mod[20] = coef_mod[18]
                coef_mod[21] = coef_mod[19]
                coef_mod[22] = 1.
                coef_mod[23] = coef_mod[22]
                coef_mod[25] = coef_mod[24]
                coef_mod[26] = coef_mod[18]
                #
                coef_mod[29] = coef_mod[27]
                coef_mod[30] = coef_mod[28]
                coef_mod[31] = 1.
                coef_mod[32] = coef_mod[31]
                coef_mod[34] = coef_mod[33]
                coef_mod[35] = coef_mod[27]
                f = fyf.yld2013_xp(sig, coef_mod, m, NT)
        elif 'YLD2013_Xp' == model and len(coef) == 200:
            f = fyf.yld2013_xp(sig, coef, m, NT)
        elif 'KB' == model and len(coef) == 6:
            f = fyf.kb(sig, coef, m)
        elif 'YLD2004NORT_21p' == model and len(coef) == 21:
            f = fyf.yld2004nort_21p(sig, coef, m)
        else:
            sys.exit("A non-compatible choice! Exiting...")
        return f - y0
    
    
    else: # dofort == False, no need to use after having Fortran implementations
        if model == 'AXIFACET':
            phi = 0.
            for i in range(Ncones):
                phi += (coef[2*i+1]*np.sqrt(sig[0]**2+sig[1]**2+2*sig[3]**2+2*sig[4]**2+2*sig[5]**2)
                - coef[2*i]*sig[2])**m
            return phi**(1./m) - 1.
    
        elif model == 'YLD2004_27p':    

            C12, C13, C21, C23, C31, C32, C44, C55, C66, 
            D12, D13, D21, D23, D31, D32, D44, D55, D66, 
            E12, E13, E21, E23, E31, E32, E44, E55, E66 = coef

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            D  = np.array([[  0., -D12, -D13,  0.,  0.,  0.],
                           [-D21,   0., -D23,  0.,  0.,  0.],
                           [-D31, -D32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., D44,  0.,  0.],
                           [  0.,   0.,   0.,  0., D55,  0.],
                           [  0.,   0.,   0.,  0.,  0., D66]])

            E  = np.array([[  0., -E12, -E13,  0.,  0.,  0.],
                           [-E21,   0., -E23,  0.,  0.,  0.],
                           [-E31, -E32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., E44,  0.,  0.],
                           [  0.,   0.,   0.,  0., E55,  0.],
                           [  0.,   0.,   0.,  0.,  0., E66]])

            s1 = np.dot(C,np.dot(T,sig))
            s2 = np.dot(D,np.dot(T,sig))
            s3 = np.dot(E,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            s22 = np.array([[s2[0], s2[5], s2[4]],
                            [s2[5], s2[1], s2[3]],
                            [s2[4], s2[3], s2[2]]])

            s33 = np.array([[s3[0], s3[5], s3[4]],
                            [s3[5], s3[1], s3[3]],
                            [s3[4], s3[3], s3[2]]])

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)
            S3, _ = np.linalg.eigh(s33)

            phi = 0.
            for i in range(3):
                for j in range(3):
                    phi += (np.abs(S1[i]-S2[j]))**m 
            phi += (np.abs(S3[0]-S3[1]))**m + (np.abs(S3[0]-S3[2]))**m + (np.abs(S3[1]-S3[2]))**m

            return (phi/6.)**(1./m) - y0

        elif model == 'YLD2004_27p_axi':    

            C12, C13, C31, C44, D12, D13, D31, D44, E12, E13, E31, E44 = coef
            # not reduced coeeficients, see above fortran implementations

            C21 = C12
            C23 = C13
            C32 = C31
            C55 = C44
            C66 = C12

            D21 = D12
            D23 = D13
            D32 = D31
            D55 = D44
            D66 = D12

            E21 = E12
            E23 = E13
            E32 = E31
            E55 = E44
            E66 = E12

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            D  = np.array([[  0., -D12, -D13,  0.,  0.,  0.],
                           [-D21,   0., -D23,  0.,  0.,  0.],
                           [-D31, -D32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., D44,  0.,  0.],
                           [  0.,   0.,   0.,  0., D55,  0.],
                           [  0.,   0.,   0.,  0.,  0., D66]])

            E  = np.array([[  0., -E12, -E13,  0.,  0.,  0.],
                           [-E21,   0., -E23,  0.,  0.,  0.],
                           [-E31, -E32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., E44,  0.,  0.],
                           [  0.,   0.,   0.,  0., E55,  0.],
                           [  0.,   0.,   0.,  0.,  0., E66]])


            s1 = np.dot(C,np.dot(T,sig))
            s2 = np.dot(D,np.dot(T,sig))
            s3 = np.dot(E,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            s22 = np.array([[s2[0], s2[5], s2[4]],
                            [s2[5], s2[1], s2[3]],
                            [s2[4], s2[3], s2[2]]])

            s33 = np.array([[s3[0], s3[5], s3[4]],
                            [s3[5], s3[1], s3[3]],
                            [s3[4], s3[3], s3[2]]])

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)
            S3, _ = np.linalg.eigh(s33)

            phi = 0.
            for i in range(3):
                for j in range(3):
                    phi += (np.abs(S1[i]-S2[j]))**m 
            phi += (np.abs(S3[0]-S3[1]))**m + (np.abs(S3[0]-S3[2]))**m + (np.abs(S3[1]-S3[2]))**m

            return (phi/6.)**(1./m) - y0

        elif model == 'YLD2004_18p':    
            C21, C23, C31, C32, C44, C55, C66, D12, D13, D21, D23, D31, D32, D44, D55, D66 = coef
            C12, C13 = 1., 1. # as suggested by van den Boogaard (2016)

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            D  = np.array([[  0., -D12, -D13,  0.,  0.,  0.],
                           [-D21,   0., -D23,  0.,  0.,  0.],
                           [-D31, -D32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., D44,  0.,  0.],
                           [  0.,   0.,   0.,  0., D55,  0.],
                           [  0.,   0.,   0.,  0.,  0., D66]])

            s1 = np.dot(C,np.dot(T,sig))
            s2 = np.dot(D,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            s22 = np.array([[s2[0], s2[5], s2[4]],
                            [s2[5], s2[1], s2[3]],
                            [s2[4], s2[3], s2[2]]])

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)

            phi = 0.
            for i in range(3):
                for j in range(3):
                   phi += (np.abs(S1[i]-S2[j]))**m 

            return (phi/4.)**(1./m) - y0

        elif model == 'YLD2004_18p_axi':    

            C12, C13, C31, C44, D12, D13, D31, D44 = coef
            # coefficients not reduced, see fortran implementation

            C21 = C12
            C23 = C13
            C32 = C31
            C55 = C44
            C66 = C12

            D21 = D12
            D23 = D13
            D32 = D31
            D55 = D44
            D66 = D12

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            D  = np.array([[  0., -D12, -D13,  0.,  0.,  0.],
                           [-D21,   0., -D23,  0.,  0.,  0.],
                           [-D31, -D32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., D44,  0.,  0.],
                           [  0.,   0.,   0.,  0., D55,  0.],
                           [  0.,   0.,   0.,  0.,  0., D66]])

            s1 = np.dot(C,np.dot(T,sig))
            s2 = np.dot(D,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            s22 = np.array([[s2[0], s2[5], s2[4]],
                            [s2[5], s2[1], s2[3]],
                            [s2[4], s2[3], s2[2]]])

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)

            phi = 0.
            for i in range(3):
                for j in range(3):
                   phi += (np.abs(S1[i]-S2[j]))**m 

            return (phi/4.)**(1./m) - y0

        elif model == 'KB':

            c, alp1, alp2, gm1, gm2, gm3 = coef
            bet1 = (alp2-alp1-1.)/2.
            bet2 = (alp1-alp2-1.)/2.
            bet3 = (1.-alp1-alp2)/2.

            C  = c*np.array([[  1., bet1, bet2,  0.,  0.,  0.],
                             [bet1, alp1, bet3,  0.,  0.,  0.],
                             [bet2, bet3, alp2,  0.,  0.,  0.],
                             [  0.,   0.,   0., gm1,  0.,  0.],
                             [  0.,   0.,   0.,  0., gm2,  0.],
                             [  0.,   0.,   0.,  0.,  0., gm3]])

            s1 = np.dot(C,sig)

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            S1, _ = np.linalg.eigh(s11)

            phi = (np.abs(S1[0]))**m + (np.abs(S1[1]))**m + (np.abs(S1[2]))**m 
            xi = (2**m+2)/(3**m)
            return (phi/xi)**(1./m) - y0

        elif model == 'YLD2004_09p':

            C12, C13, C21, C23, C31, C32, C44, C55, C66 = coef

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            s1 = np.dot(C,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            S1, _ = np.linalg.eigh(s11)

            phi = 0.
            for i in range(3):
                for j in range(i):
                    phi += (np.abs(S1[i]-S1[j]))**m 
                    
            return (phi/2.)**(1./m) - y0

        elif model == 'YLD2004_09p_axi':

            C12, C13, C31, C44 = coef

            C21 = C12
            C23 = C13
            C32 = C31
            C55 = C44
            C66 = C12

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            s1 = np.dot(C,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            S1, _ = np.linalg.eigh(s11)

            phi = 0.
            for i in range(3):
                for j in range(i):
                    phi += (np.abs(S1[i]-S1[j]))**m 

            return (phi/2.)**(1./m) - y0

        elif model == 'YLD2004_09p_axi_notrace':

            C12, C13, C44 = coef

            C31 = 2*C13 - C12

            C21 = C12
            C23 = C13
            C32 = C31
            C55 = C44
            C66 = C12

            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[  0., -C12, -C13,  0.,  0.,  0.],
                           [-C21,   0., -C23,  0.,  0.,  0.],
                           [-C31, -C32,   0.,  0.,  0.,  0.],
                           [  0.,   0.,   0., C44,  0.,  0.],
                           [  0.,   0.,   0.,  0., C55,  0.],
                           [  0.,   0.,   0.,  0.,  0., C66]])

            s1 = np.dot(C,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            S1, _ = np.linalg.eigh(s11)

            phi = 0.
            for i in range(3):
                for j in range(i):
                    phi += (np.abs(S1[i]-S1[j]))**m 

            return (phi/2.)**(1./m) - y0

        elif model == 'YLD2004_09p_ortho':

            a, b, c, f, g, h = coef
            
            T = np.array([[ 2., -1., -1., 0., 0., 0.],
                          [-1.,  2., -1., 0., 0., 0.],
                          [-1., -1.,  2., 0., 0., 0.],
                          [ 0.,  0.,  0., 3., 0., 0.],
                          [ 0.,  0.,  0., 0., 3., 0.],
                          [ 0.,  0.,  0., 0., 0., 3.]])/3.

            C  = np.array([[ b+c,   -c,   -b,  0.,  0.,  0.],
                           [  -c,  c+a,   -a,  0.,  0.,  0.],
                           [  -b,   -a,  a+b,  0.,  0.,  0.],
                           [  0.,   0.,   0., 3*f,  0.,  0.],
                           [  0.,   0.,   0.,  0., 3*g,  0.],
                           [  0.,   0.,   0.,  0.,  0., 3*h]])/3.

            s1 = np.dot(C,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[5], s1[4]],
                            [s1[5], s1[1], s1[3]],
                            [s1[4], s1[3], s1[2]]])

            S1, _ = np.linalg.eigh(s11)

            phi = (np.abs(S1[0]-S1[1]))**m + (np.abs(S1[0]-S1[2]))**m + (np.abs(S1[1]-S1[2]))**m 

            return (phi/2.)**(1./m) - y0

        elif model == 'Hill48':
            F, H, L = coef
            G = F
            M = L
            N = F + 2.*H
            SIG = k*SIG
            phi = np.sqrt(F*(SIG[1,1]-SIG[2,2])**2+G*(SIG[2,2]-SIG[0,0])**2+H*(SIG[0,0]-SIG[1,1])**2+2.*L*SIG[1,2]**2+2.*M*SIG[2,0]**2+2.*N*SIG[0,1]**2)
            return phi - y0

        elif model == 'Yld2000':

            C11, C22, C66, D11, D12, D21, D22, D66 = coef

            sig = k * np.array([SIG[1,1], SIG[2,2], SIG[1,2]])

            T = np.array([[ 2., -1., 0.],
                          [-1.,  2., 0.],
                          [ 0.,  0., 3.]])/3.

            C1 = np.array([[  C11, 0., 0.],
                           [  0., C22, 0.],
                           [  0., 0., C66]])

            C2 = np.array([[ D11,  D12, 0.],
                           [ D21,  D22, 0.],
                           [ 0.,  0., D66]])

            s1 = np.dot(C1,np.dot(T,sig))
            s2 = np.dot(C2,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[2]],
                            [s1[2], s1[1]]])

            s22 = np.array([[s2[0], s2[2]],
                            [s2[2], s2[1]]])    

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)


            phi = (np.abs(S1[0]-S1[1]))**m + (np.abs(2*S2[0]+S2[1]))**m + (np.abs(2*S2[1]+S2[0]))**m

            return (phi/2.)**(1./m) - y0

        elif model == 'Yld2000alph':

            alp1, alp2, alp3, alp4, alp5, alp6, alp7, alp8 = coef

            sig = k * np.array([SIG[1,1], SIG[2,2], SIG[1,2]])

            T = np.array([[ 2., -1., 0.],
                          [-1.,  2., 0.],
                          [ 0.,  0., 3.]])/3.

            C1 = np.array([[  alp1,  0.,  0.],
                           [   0., alp2,  0.],
                           [   0.,  0., alp7]])

            C2 = np.array([[ 4.*alp5-alp3,  2.*(alp6-alp4),  0.],
                           [ 2.*(alp3-alp5),  4.*alp4-alp6,  0.],
                           [       0.,              0.,    3.*alp8]])*1./3.

            s1 = np.dot(C1,np.dot(T,sig))
            s2 = np.dot(C2,np.dot(T,sig))

            s11 = np.array([[s1[0], s1[2]],
                            [s1[2], s1[1]]])

            s22 = np.array([[s2[0], s2[2]],
                            [s2[2], s2[1]]])    

            S1, _ = np.linalg.eigh(s11)
            S2, _ = np.linalg.eigh(s22)


            phi = (np.abs(S1[0]-S1[1]))**m + (np.abs(2*S2[0]+S2[1]))**m + (np.abs(2*S2[1]+S2[0]))**m

            return (phi/2.)**(1./m) - y0

        elif model == 'Yld89':

            a, c, h, p = coef

            sig = k * np.array([SIG[1,1], SIG[2,2], SIG[1,2]])

            C1 = np.array([[   1., 0., 0.],
                           [   0., h,  0.],
                           [   0., 0., p]])


            s1 = np.dot(C1,sig)

            s11 = np.array([[s1[0], s1[2]],
                            [s1[2], s1[1]]])  

            S1, _ = np.linalg.eigh(s11)


            phi = a*(np.abs(S1[1]))**m + a*(np.abs(S1[0]))**m + c*(np.abs(S1[1]-S1[0]))**m

            return (phi/2.)**(1./m) - y0

        elif model == 'BBC2003':

            L, M, N, P, Q, R, S, T = coef
            #alph = 1./2.

            sig = k * np.array([SIG[1,1], SIG[2,2], SIG[1,2]])

            GM  = (L*sig[0]+M*sig[1])/2.
            PSI = np.sqrt(((N*sig[0]-P*sig[1])/2.)**2 + (Q*sig[2])**2)
            LAM = np.sqrt(((R*sig[0]-S*sig[1])/2.)**2 + (T*sig[2])**2)

            phi = (np.abs(GM+PSI))**m + (np.abs(GM-PSI))**m + (np.abs(2*LAM))**m

            return (phi/2.)**(1./m) - y0
        
        else:
            sys.exit("A non-compatible choice! Exiting...")

        
def get_coef_axisym_fort(coefa, model, NT, sigY):
#     coef = coefa
    coef = coefa[:-1]
    a = coefa[-1] 
    coef_mod = np.zeros(200)
    if model == 'YLD2004_09p' and len(coef) == 3:
        # C12, C13, C44
        # C31 = 1
        coef_mod[[0,1,6]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[4] = 1.
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
    elif model[-3:] == '18p' and len(coef) == 7:
        # C12, C13, C31, C44, D12, D13, D44
        # D31 = 1 
        coef_mod[[0,1,4,6,9,10,15]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
        coef_mod[11] = coef_mod[9]
        coef_mod[12] = coef_mod[10]
        coef_mod[13] = 1.
        coef_mod[14] = coef_mod[13]
        coef_mod[16] = coef_mod[15]
        coef_mod[17] = coef_mod[9]
    elif model in ['YLD2004_27p', 'YLD2013_27p'] and len(coef) == 10:
        # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44
        # D31 = 1
        # E31 = 1
        coef_mod[[0,1,4,6,9,10,15,18,19,24]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
        coef_mod[11] = coef_mod[9]
        coef_mod[12] = coef_mod[10]
        coef_mod[13] = 1.
        coef_mod[14] = coef_mod[13]
        coef_mod[16] = coef_mod[15]
        coef_mod[17] = coef_mod[9]
        coef_mod[20] = coef_mod[18]
        coef_mod[21] = coef_mod[19]
        coef_mod[22] = 1.
        coef_mod[23] = coef_mod[22]
        coef_mod[25] = coef_mod[24]
        coef_mod[26] = coef_mod[18]
    elif model == 'YLD2011_27p' and len(coef) == 11:
        # C12, C13, C31, C44, D12, D13, D44, E12, E13, E31, E44
        # D31 = 1
        coef_mod[[0,1,4,6,9,10,15,18,19,22,24]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
        coef_mod[11] = coef_mod[9]
        coef_mod[12] = coef_mod[10]
        coef_mod[13] = 1.
        coef_mod[14] = coef_mod[13]
        coef_mod[16] = coef_mod[15]
        coef_mod[17] = coef_mod[9]
        coef_mod[20] = coef_mod[18]
        coef_mod[21] = coef_mod[19]
        coef_mod[23] = coef_mod[22]
        coef_mod[25] = coef_mod[24]
        coef_mod[26] = coef_mod[18]
    elif model == 'YLD2013_36p' and len(coef) == 13:
        # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44, F12, F13, F44
        # D31 = 1
        # E31 = 1
        # F31 = 1
        NT = 4
        model = 'YLD2013_Xp'
        coef_mod = np.zeros(200)
        coef_mod[[0,1,4,6,9,10,15,18,19,24,27,28,33]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
        #
        coef_mod[11] = coef_mod[9]
        coef_mod[12] = coef_mod[10]
        coef_mod[13] = 1.
        coef_mod[14] = coef_mod[13]
        coef_mod[16] = coef_mod[15]
        coef_mod[17] = coef_mod[9]
        #
        coef_mod[20] = coef_mod[18]
        coef_mod[21] = coef_mod[19]
        coef_mod[22] = 1.
        coef_mod[23] = coef_mod[22]
        coef_mod[25] = coef_mod[24]
        coef_mod[26] = coef_mod[18]
        #
        coef_mod[29] = coef_mod[27]
        coef_mod[30] = coef_mod[28]
        coef_mod[31] = 1.
        coef_mod[32] = coef_mod[31]
        coef_mod[34] = coef_mod[33]
        coef_mod[35] = coef_mod[27]
    elif model == 'YLD2013_45p' and len(coef) == 16:
        # C12, C13, C31, C44, D12, D13, D44, E12, E13, E44, F12, F13, F44, G12, G13, G44
        # D31 = 1
        # E31 = 1
        # F31 = 1
        # G31 = 1
        NT = 5
        model = 'YLD2013_Xp'
        coef_mod = np.zeros(200)
        coef_mod[[0,1,4,6,9,10,15,18,19,24,27,28,33,36,37,42]] = coef
        coef_mod[2] = coef_mod[0]
        coef_mod[3] = coef_mod[1]
        coef_mod[5] = coef_mod[4]
        coef_mod[7] = coef_mod[6]
        coef_mod[8] = coef_mod[0]
        #
        coef_mod[11] = coef_mod[9]
        coef_mod[12] = coef_mod[10]
        coef_mod[13] = 1.
        coef_mod[14] = coef_mod[13]
        coef_mod[16] = coef_mod[15]
        coef_mod[17] = coef_mod[9]
        #
        coef_mod[20] = coef_mod[18]
        coef_mod[21] = coef_mod[19]
        coef_mod[22] = 1.
        coef_mod[23] = coef_mod[22]
        coef_mod[25] = coef_mod[24]
        coef_mod[26] = coef_mod[18]
        #
        coef_mod[29] = coef_mod[27]
        coef_mod[30] = coef_mod[28]
        coef_mod[31] = 1.
        coef_mod[32] = coef_mod[31]
        coef_mod[34] = coef_mod[33]
        coef_mod[35] = coef_mod[27]
        #
        coef_mod[38] = coef_mod[36]
        coef_mod[39] = coef_mod[37]
        coef_mod[40] = 1.
        coef_mod[41] = coef_mod[40]
        coef_mod[43] = coef_mod[42]
        coef_mod[44] = coef_mod[36]
    else:
        sys.exit("A non-compatible choice! Exiting...")
        
    res = fyf.py2f_yldfun.get_residual(coef_mod, model, a, NT, sigY)
    return res


def genStresses(N, yfun, plot_axes, save2file=False, dofort=True):
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
    try:
        abs_comp = [plot_axes[2]]
        zij = [int(x)-1 for x in plot_axes[2]]
        zij.sort()
        zij = tuple(zij)
        zi, zj = zij
        if zij in [xij, yij]:
            sys.exit('Out-of-plane stress component must differ.')
        elif zij not in normal_components + shear_components:
            sys.exit('Out-of-plane stress component is NA.')
        s0_list = plot_axes[3]
    except:
        zij = None
        abs_comp = None
        s0_list = [0.]

    angles = np.linspace(0, 2.*np.pi, N)
            
    s = np.zeros((3,3,N))
    stresses = np.zeros((3,3,N*len(s0_list)))
    s[xi,xj,:] = np.cos(angles)
    s[yi,yj,:] = np.sin(angles)
    # case of shear
    if xi != xj:
        s[xj,xi,:] = np.cos(angles)
    if yi != yj:
        s[yj,yi,:] = np.sin(angles)

    j = 0
    nosol = 0
    for s0 in s0_list:
        if zij is not None:
            s[zi,zj,:] = s0
            if zi != zj:
                s[zj,zi,:] = s0

        for i in range(N):
            if i == 0:
                x0 = yfun['y0']
            else:
                x0 = k
            k, infodict, ier, mesg = fsolve(YF, x0=x0, args=(s[:,:,i], yfun, abs_comp, dofort), full_output=True)
            if ier == 1:
                stresses[:,:,j] = k*s[:,:,i]
                if zij is not None:
                    stresses[zi,zj,j] = s0
                    if zi != zj:
                        stresses[zj,zi,j] = s0
                j += 1
            else:
                nosol += 1

    stresses = stresses[:,:,:j]
    if nosol > 0:
        print('Number of not converged stresses: {}'.format(nosol))
    
    if type(save2file) is str:
        try:
            np.save(save2file, stresses)
        except:
            print('Cannot save the file.')
            
    return stresses


def Barlat2SMM(coef, model):
    
    if model == 'YLD2004_18p':
        C21, C31, C23, C32, C44, C55, C66, D12, D21, D13, D31, D23, D32, D44, D55, D66 = coef
        C12, C13 = 1., 1.
    elif model == 'YLD2004_18p_axi':
        C12, C13, C31, C44, D12, D13, D44 = coef
        D31 = 1.
    elif model == 'YLD2004_09p':
        C12, C21, C13, C31, C23, C32, C44, C55, C66 = coef
    elif model == 'YLD2004_09p_axi':
        C12, C13, C31, C44 = coef

    C21 = C12
    C23 = C13
    C32 = C31
    C55 = C44
    C66 = C12
    
    try:
        D21 = D12
        D23 = D13
        D32 = D31
        D55 = D44
        D66 = D12
    except:
        pass
        
    if model in ['YLD2004_18p', 'YLD2004_18p_axi']:
        return [C12, C13, C21, C23, C31, C32, C66, C44, C55, D12, D13, D21, D23, D31, D32, D66, D44, D55]
    elif model in ['YLD2004_09p','YLD2004_09p_axi']:
        return [C12, C13, C21, C23, C31, C32, C66, C44, C55]
    
def SMM2Barlat(coef, model):
    if model == 'YLD2004_18p':
        C12, C13, C21, C23, C31, C32, C66, C44, C55, D12, D13, D21, D23, D31, D32, D66, D44, D55 = coef
        coefB = [C21, C31, C23, C32, C44, C55, C66, D12, D21, D13, D31, D23, D32, D44, D55, D66]
    elif model == 'YLD2004_18p_axi':
        C12, C13, C21, C23, C31, C32, C66, C44, C55, D12, D13, D21, D23, D31, D32, D66, D44, D55 = coef
        if C12 != C21 or C12 != C66 or C44 != C55 or D12 != D21 or D44 != D55:
            print('Not axisymmetrical coefficients!!!')
        coefB = [C12, C13, C31, C44, D12, D13, D44]
    elif model == 'YLD2004_09p':
        C12, C13, C21, C23, C31, C32, C66, C44, C55 = coef
        coefB = [C12, C21, C13, C31, C23, C32, C44, C55, C66]
    elif model == 'YLD2004_09p_axi':
        C12, C13, C21, C23, C31, C32, C66, C44, C55 = coef
        if C12 != C21 or C12 != C66 or C44 != C55:
            print('Not axisymmetrical coefficients!!!')
        coefB = [C12, C13, C31, C44]       
    return coefB
    
    
def testAxiSym(yfun, flag='test'):
    f = []
    P = np.linspace(0.,2.*np.pi, 100)
    if flag=='test':
        SIG0 = np.random.rand(3,3)
        SIG0[1,0] = SIG0[0,1]
        SIG0[2,0] = SIG0[0,2]
        SIG0[1,2] = SIG0[2,1]
    
        for p in P:
            R = np.array([[ np.cos(p), np.sin(p), 0.],
                          [-np.sin(p), np.cos(p), 0.],
                          [ 0.,        0,         1.]])
            SIG = np.dot(R.T,np.dot(SIG0, R))
            f.append(YF(1., SIG, yfun))
        fig, ax = plt.subplots()
        ax.plot(P, f)
    
    elif flag=='plotxy':
        SIG = np.zeros((3,3))
        plxy = np.zeros((100,2))
        for i, p in enumerate(P):
            SIG[0,0], SIG[2,2] = np.cos(p), np.sin(p)            
            k, infodict, ier, mesg = fsolve(YF,x0=1.,args=(SIG, coef, m, model), full_output=True)
            if ier == 1:
                SIGk = k*SIG
                plxy[i,:] = [SIGk[0,0], SIGk[2,2]]
            else:
                print('No solution found')
        fig, ax = plt.subplots()
        ax.plot(plxy[:,0],plxy[:,1])
        return plxy
    
    
def testMin(yfun, y0, stresses, N=100):
    f = get_coef(yfun['coef'], stresses, yfun, y0)
    eps = 1e-5
    for i in range(N): 
        M = np.random.randint(len(yfun['coef']))
        random_choice = np.arange(len(yfun['coef']))
        np.random.shuffle(random_choice)
        random_choice = random_choice[:M]
        c = yfun['coef'].copy()
        c[random_choice] += eps
        g = get_coef(c, stresses, yfun, y0)
        if g < f:
            print('Not a minimm: {}'.format(g-f))
    
def VM(s):
    return np.sqrt(0.5*( (s[0,0]-s[1,1])**2 + (s[0,0]-s[2,2])**2 + (s[1,1]-s[2,2])**2 + 6.*(s[0,1]**2 + s[0,2]**2 + s[1,2]**2) ))

def calcDpart(start_col, end_col, stresses, yfun):
    yfun0 = deepcopy(yfun)
    NT = yfun0['Ntransf']
    dc = 1e-6
    N = stresses.shape[2]
    if 'YLD2013_Xp' in yfun0['func']:
        c0 = yfun0['coef'][:9*NT]
    else:
        c0 = yfun0['coef']
    yfun['y0'] = 1.
    y0 = yfun['y0']
    
    n = 0
    D = np.zeros((end_col-start_col, N))
    for i in range(start_col, end_col):
        c = c0.copy()
        c[i] = c0[i] + dc

        if 'YLD2013_Xp' in yfun['func']:
            yfun['coef'][:9*NT] = c
        else:
            yfun['coef'] = c

        for j in range(N):
            s = stresses[:,:,j]
#            k, infodict, ier, mesg = fsolve(YF, x0=y0, args=(s, yfun, y0, keep, dofort), full_output=True)
            k = y0/(YF(y0, s, yfun) + y0)
#            D[i,j] = (VM(k*s) - r[j])/dc
            D[n,j] = YF(k, s, yfun0)/dc

        n += 1
    return D

def getJac(stresses, yfun, doparallel):
    t0 = time.time()
    yfun0 = deepcopy(yfun)
    NT = yfun0['Ntransf']
    dc = 1e-6
    N = stresses.shape[2]
    if 'YLD2013_Xp' in yfun0['func']:
        c0 = yfun0['coef'][:9*NT]
    else:
        c0 = yfun0['coef']
    yfun['y0'] = 1.
    y0 = yfun['y0']
#    keep=None
#    r = np.zeros(N)
#     for j in range(N):
#         s = stresses[:,:,j]
#         k, infodict, ier, mesg = fsolve(YF, x0=y0, args=(s, yfun, y0, keep, dofort), full_output=True)
#         if ier == 1:
#             r[j] = VM(k*s)
#         else:
#             print('No solution found')
        
    Ncoef = len(c0)
    D = np.zeros((len(c0), N))
    if not doparallel:
        for i in range(Ncoef):
#             print(i)
            c = c0.copy()
            c[i] = c0[i] + dc

            if 'YLD2013_Xp' in yfun['func']:
                yfun['coef'][:9*NT] = c
            else:
                yfun['coef'] = c
                
            for j in range(N):
                s = stresses[:,:,j]
    #            k, infodict, ier, mesg = fsolve(YF, x0=y0, args=(s, yfun, y0, keep, dofort), full_output=True)
                k = y0/(YF(y0, s, yfun) + y0)
    #            D[i,j] = (VM(k*s) - r[j])/dc
                D[i,j] = YF(k, s, yfun0)/dc
        
    else: # if parallel
        Ncpus = mp.cpu_count()
        start = np.linspace(0,Ncoef-Ncpus,Ncpus,dtype=int)
        stop = np.append(start[1:], Ncoef)
        pool = mp.Pool(Ncpus)
        results = pool.starmap(calcDpart, [(start[i], stop[i], stresses, yfun) for i in range(Ncpus)])
        for i, r in enumerate(results):
            D[start[i]:stop[i],:] = r
   
    S = D @ D.T
    lam, P = np.linalg.eigh(S)
    print(f'PCA took: {time.time()-t0} s.')
    yfun = deepcopy(yfun0)
    return lam, P
    
def get_coefa(coefs, model, NT, y0):
    coef = coefs[:-1]
    a = coefs[-1]
    coef_mod = np.zeros(200)
    if model == 'YLD2004_09p':
        # C12 = C13 = 1 
        # C21, C23, C31, C32, C44, C55, C66
        coef_mod[:2] = 1.
        coef_mod[2:9] = coef
    elif model == 'YLD2004_09p_ambig':
        coef_mod[:9] = coef
    elif model == 'YLD2004_18p':
        # C12 = C13 = 1 
        # C21, C23, C31, C32, C44, C55, C66,
        # D21, D23, D31, D32, D44, D55, D66
        coef_mod[:2] = 1.
        coef_mod[2:18] = coef
    elif 'YLD2011_18p' in model:
        coef_mod[:2] = 1.
        coef_mod[2:18] = coef
    elif 'YLD2004_27p' in model:
        coef_mod[:2] = 1.
        coef_mod[2:18] = coef[:16]
        coef_mod[18:20] = 1.
        coef_mod[20:27] = coef[16:]
    elif 'YLD2013_Xp' == model:
        if NT == 1:
            coef_mod[:9] = coef
    elif 'KB' == model:
        coef_mod[:6] = coef
    elif 'YLD2004NORT_21p' == model:
        coef_mod[:2] = 1.
        coef_mod[2:9] = coef[:7]
        coef_mod[9:12] = 0.
        coef_mod[12:21] = coef[7:]
    else:
        sys.exit('A non-compatible choice! Exiting...')
        
    res = fyf.py2f_yldfun.get_residual(coef_mod, model, a, NT, y0)
    return res

def get_coef(coef, model, a, NT, y0):
    coef_mod = np.zeros(200)
    if 'YLD2004_09p' in model:
        if model == 'YLD2004_09p_ambig':
            coef_mod[:9] = coef
        elif model == 'YLD2004_09p':
            # C12 = C13 = 1 
            # C21, C23, C31, C32, C44, C55, C66
#
#         coef_mod[0] = coef[0]+coef[2]-coef[3]
#         coef_mod[1] = coef[0]-coef[1]+coef[2]
#         coef_mod[2:9] = coef
            coef_mod[:2] = 1.
            coef_mod[2:9] = coef
            
        res = fyf.py2f_yldfun.get_residual(coef_mod, 'YLD2004_09p', a, NT, y0)

    elif 'YLD2004_18p' in model:
        if model == 'YLD2004_18p_ambig':
            coef_mod[:18] = coef
        elif model == 'YLD2004_18p':
            coef_mod[:2] = 1.
            coef_mod[2:18] = coef
        elif model == 'YLD2004nopres_18p':
            # 4 contraints - 14 indep params
            # C31 = C13 + C23 - C21
            # C32 = C13 + C23 - C12
            # D31 = D13 + D23 - D21
            # D32 = D13 + D23 - D12
            coef_mod[:4] = coef[:4]
            coef_mod[4] = coef[1] + coef[3] - coef[2]
            coef_mod[5] = coef[1] + coef[3] - coef[0]
            coef_mod[6:13] = coef[4:11]
            coef_mod[13] = coef[8] + coef[10] - coef[9]
            coef_mod[14] = coef[8] + coef[10] - coef[7]
            coef_mod[15:18] = coef[11:14]
        elif model == 'YLD2004indep_18p':
            # 2 contraints - 16 indep params
            # C31 = C13 + C23 - C21
            # C32 = C13 + C23 - C12
            coef_mod[:4] = coef[:4]
            coef_mod[4] = coef[1] + coef[3] - coef[2]
            coef_mod[5] = coef[1] + coef[3] - coef[0]
            coef_mod[6:18] = coef[4:16]
            
        res = fyf.py2f_yldfun.get_residual(coef_mod, 'YLD2004_18p', a, NT, y0)
    elif 'YLD2011_18p' in model:
        coef_mod[:2] = 1.
        coef_mod[2:18] = coef
        res = fyf.py2f_yldfun.get_residual(coef_mod, 'YLD2011_18p', a, NT, y0)
    else:
        print('A non-compatible choice! Exiting...')
        res = None
    return res

def get_coef_extra_weight(coef, model, a, NT, y0, weight):
    coef_mod = np.zeros(200)
    if model == 'YLD2004indep_18p':
        # 2 contraints - 16 indep params
        # C31 = C13 + C23 - C21
        # C32 = C13 + C23 - C12
        coef_mod[:4] = coef[:4]
        coef_mod[4] = coef[1] + coef[3] - coef[2]
        coef_mod[5] = coef[1] + coef[3] - coef[0]
        coef_mod[6:18] = coef[4:16]
    else:
        print('A non-compatible choice! Exiting...')
        
    res = fyf.py2f_yldfun.get_residual_extra_weight(coef_mod, 'YLD2004_18p', a, NT, y0, weight)
    return res

def get_coef_degen(coef, model, a, NT, y0, K, fixID):
    coef_mod = np.zeros(200)
    coef_mod[[i for i in range(9*NT) if i not in fixID]] = coef
    for i in range(len(K)):
        coef_mod[fixID[i]] = K[i]
    res = fyf.py2f_yldfun.get_residual(coef_mod, model, a, NT, y0)
    return res