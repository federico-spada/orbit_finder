import numpy as np
from orbit_finder import *

import config as cf

# parameters in Marsden+73 g(r) function
alpha, r0, nm, nn, nk = 0.1113, 2.808, 2.15, 5.093, 4.6142


### FUNCTION IMPLEMENTING THE NON-GRAVITATIONAL ACCELERATION MODEL ------------
# reorganized for clarity; fitting of DT will have to be added back later
def NonGravAccel(r_, v_, parms_):
    r = np.linalg.norm(r_)
    # (u1_, u2_, u3_): radial/transverse/normal unit vectors
    u1_ = r_/r
    u3_ = np.cross(r_,v_)/np.linalg.norm(np.cross(r_,v_))
    u2_ = np.cross(u3_,u1_)
    u_ = np.c_[u1_, u2_, u3_]
    # Marsden+73 g(r) function 
    g = alpha * (r/r0) ** -nm * (1. + (r/r0) ** nn ) ** -nk
    # construct acceleration vector and its derivatives
    aNG_ = np.zeros(3)
    dadp = np.zeros((3, len(parms_)))
    for i, Ai in enumerate(parms_):
        aNG_ = aNG_ + g * Ai * u_[:,i]
        dadp[:,i] = g * u_[:,i]
    return aNG_, dadp
