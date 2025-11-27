import numpy as np
from orbit_finder import *
import config as cf

### FUNCTION IMPLEMENTING THE NON-GRAVITATIONAL ACCELERATION MODEL ------------
def NonGravAccel(r_, v_, parms_, nongrav_coeffs):
    # >>> cross product matrix (cf. Montenbruck & Gill 2000) 
    def X(w_):
        wx, wy, wz = w_
        return np.array([ [0., -wz, +wy], [+wz, 0., -wx], [-wy, +wx, 0.] ])
    # <<<
    alpha, r0, m, n, k = nongrav_coeffs
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
    # also arranged as lists for later convenience:
    dudr = [du1dr, du2dr, du3dr]
    dudv = [du1dv, du2dv, du3dv]
    # Marsden+73 g(r) function, symmetric (no delay parameter)
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
        Ai = parms_[i]
        aNG_       += g * Ai * u_[:,i]
        dadpNG[:,i] = g * u_[:,i]
        dadrNG     += g * Ai * dudr[i] + dgdr * Ai * np.outer(u_[:,i], u1_)
        dadvNG     += g * Ai * dudv[i]
    if n_p == 4:
        # account for emission peak delay parameter DT
        A1, A2, A3, DT = parms_ # 3*[au/d^2], [days]
        Au_ = A1 * u1_ + A2 * u2_ + A3 * u3_
        r1_ = spice.prop2b(cf.MU_S, np.r_[r_, v_], DT)[:3]
        r1 = np.linalg.norm(r1_)
        g1 = alpha * (r1/r0) ** -m * (1. + (r1/r0) ** n ) ** -k
        # acceleration (note g1 in place of g)
        aNG_ = g1 * Au_
        # derivative wrt DT
        delta = 0.05 # days
        rp = np.linalg.norm(spice.prop2b(cf.MU_S, np.r_[r_, v_], DT+delta)[:3])
        rm = np.linalg.norm(spice.prop2b(cf.MU_S, np.r_[r_, v_], DT-delta)[:3])
        gp = alpha * (rp/r0) ** -m * (1. + (rp/r0) ** n ) ** -k
        gm = alpha * (rm/r0) ** -m * (1. + (rm/r0) ** n ) ** -k
        dgdr  = (gp - gm)/(rp - rm)
        dgdDT = (gp - gm)/(2*delta)
        dadpNG[:,3] = dgdDT * Au_
        # derivatives wrt r_, v_
        dadrNG = np.zeros((3,3))
        dadvNG = np.zeros((3,3))
    return aNG_, dadrNG, dadvNG, dadpNG
