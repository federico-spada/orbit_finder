import numpy as np
from orbit_finder import *

import config as cf

# parameters in Marsden+73 g(r) function
alpha, r0, m, n, k = 0.1113, 2.808, 2.15, 5.093, 4.6142


### FUNCTION IMPLEMENTING THE NON-GRAVITATIONAL ACCELERATION MODEL ------------
def NonGravAccel(r_, v_, parms_):
    # >>> cross product matrix (cf. Montenbruck & Gill 2000) 
    def X(w_):
        wx, wy, wz = w_
        return np.array([ [0., -wz, +wy], [+wz, 0., -wx], [-wy, +wx, 0.] ])
    # <<<
    r = np.linalg.norm(r_)
    h_ = np.cross(r_, v_)
    h = np.linalg.norm(h_) 
    # (u1_, u2_, u3_): radial/transverse/normal unit vectors  
    u1_ = r_/r
    u3_ = h_/h
    u2_ = np.cross(u3_,u1_)
    u_ = np.c_[u1_, u2_, u3_]
    # their partial derivatives wrt r_, v_:
    du1dr = (np.eye(3) - np.outer(u1_, u1_))/r
    du3dr = (np.eye(3) - np.outer(u3_, u3_))/h @ (-X(v_)) 
    du2dr = X(u3_) @ du1dr - X(u1_) @ du3dr
    du1dv = np.zeros((3,3)) 
    du3dv = (np.eye(3) - np.outer(u3_, u3_))/h @ (+X(r_))
    du2dv = -X(u1_) @ du3dv
    dudr = np.array([du1dr, du2dr, du3dr])
    dudv = np.array([du1dv, du2dv, du3dv])
    # Marsden+73 g(r) function
    g = alpha * (r/r0) ** -m * (1. + (r/r0) ** n ) ** -k
    dgdr = -g/r * (m + k*n/(1 + (r0/r) ** n) )
    # construct acceleration vector and derivatives wrt Ai
    n_p = len(parms_)  
    aNG_ = np.zeros(3)
    dadpNG = np.zeros((3, n_p))
    dadrNG = np.zeros((3,3))
    dadvNG = np.zeros((3,3))
    # note numerical factors to have Ai's in units of 10^-8 au/d^2 
    for i in range(min(n_p, 3)):
        Ai = parms_[i] * 1e-8
        aNG_       += g * Ai * u_[:,i]
        dadpNG[:,i] = g * u_[:,i] * 1e-8
        dadrNG     += g * Ai * dudr[i] + dgdr * Ai * np.outer(u_[:,i], u1_)
        dadvNG     += g * Ai * dudv[i]
    return aNG_, dadrNG, dadvNG, dadpNG
