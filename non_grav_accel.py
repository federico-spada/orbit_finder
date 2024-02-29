import numpy as np
from orbit_finder import *

### FUNCTION IMPLEMENTING THE NON-GRAVITATIONAL ACCELERATION MODEL ------------
#   *** To be modified or rewritten by the user, as needed ***
#   RTN decomposition, with option for ACN (comment/uncomment appropriate lines below).
#   Required argument "parms_" can have 1 to 4 components: 
#   - If parms_ is passed with up to 3 elements, a symmetric NG acceleration model is 
#     implemented, with radial dependence according to the Marsden+73 parametrization, 
#     and the parameters A_i (up to i=3) are fitted in the orbit determination procedure. 
#   - If parms_ is passed with 4 elements, an asymmetric radial dependence is used,
#     and the perihelion offset tau (in days) is also fit.
def NonGravAccel(r_,v_,parms_):
    ### NG acceleration vector decomposition
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    # radial/transverse/normal:
    u1_ = r_/r
    # or, alternatively, along-track/cross-track/normal:
    #u1_ = v_/v
    u3_ = np.cross(r_,v_)/np.linalg.norm(np.cross(r_,v_))
    u2_ = np.cross(u3_,u1_)
    ### g function     
    # Marsden+73 parametrization     
    aa, r0, mm, nn, kk = 0.1113, 2.808, 2.15, 5.093, 4.6142
    if len(parms_) <= 3:
        # symmetric model - no perihelion offset: 
        A1, A2, A3 = np.pad(parms_, (0, 3-len(parms_)))
        A_ = A1 * u1_ + A2 * u2_ + A3 * u3_
        r = np.linalg.norm(r_)
        g = aa/(r/r0)**mm/(1. + (r/r0)**nn)**kk
        aNG_ = g * A_
        dadp = g * np.c_[u1_, u2_, u3_]
        dadp = np.delete(dadp,range(2,len(parms_)-1,-1),axis=1)
    else:
        # perihelion offset, tau, also being fitted:
        A1, A2, A3, tau = parms_
        A_ = A1 * u1_ + A2 * u2_ + A3 * u3_
        # get r', dr'/dtau
        r1, dr1dtau = KeplerUniv(r_,v_,-tau,mu_s)
        # this is g(r') = g[r(t-tau)]  
        g = aa/(r1/r0)**mm/(1. + (r1/r0)**nn)**kk
        dgdr1 = -(g/r1)*( mm*(1+(r1/r0)**nn) + kk*nn*(r1/r0)**nn )/(1 + (r1/r0)**nn)
        dgdtau = dgdr1 * dr1dtau * days # note: "days" factor for consistency of scaling!
        # NG acceleration
        aNG_ = g * A_
        # matrix with partials
        dadp = np.c_[g*u1_, g*u2_, g*u3_, dgdtau*A_]
    return aNG_, dadp

# Modified Kepler solver, universal variables formulation; useful references:
# - Section 4.3 of Bate, Mueller, White 1971; 
# - Section 4.5 of Battin 1999 (especially equations 4.81, 4.82, 4.83)
def KeplerUniv(ro_,vo_,dt,mu):
    # Stumpff function C(z)
    def C(z):
        if z > 0:
            C = (1 - np.cos(np.sqrt(z)))/z
        elif z < 0:
            C = (np.cosh(np.sqrt(-z)) - 1)/(-z)
        else:
            C = 1/2
        return C
    # Stumpff function S(z)
    def S(z):
        if z > 0:
            S = (np.sqrt(z) - np.sin(np.sqrt(z)))/np.sqrt(z)**3
        elif z < 0:
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/np.sqrt(-z)**3
        else:
            S = 1/6
        return S
    # initializations
    ro = np.linalg.norm(ro_)
    vo = np.linalg.norm(vo_)
    qo = np.dot(ro_,vo_)/np.sqrt(mu) # this is sigma_o
    alpha = 2/ro - vo**2/mu
    # initial guess for universal anomaly x
    x = np.sqrt(mu)*np.abs(alpha)*dt
    dx = 1
    # Newton iteration to find x
    while (np.abs(dx) > 1e-8):
       z = alpha*x**2
       F = x**3*S(z) + qo*x**2*C(z) + ro*x*(1-z*S(z)) - np.sqrt(mu)*dt
       dFdx = x**2*C(z) + qo*x*(1-z*S(z)) + ro*(1-z*C(z))
       dx = - F / dFdx
       x = x + dx
    r = dFdx
    q = qo*(1-z*C(z)) + (1 - alpha*ro)*x*(1-z*S(z)) # this is sigma
    drdt = np.sqrt(mu)*q/r
    return r, drdt
