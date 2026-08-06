import numpy as np
import spiceypy as spice
from astroquery.jplhorizons import Horizons
import constants as cs

mu0 = 132712440041.279419 * cs.DAYS**2 / cs.AU**3



def QueryHorizons(desig, fit_epoch):
    et0 = spice.str2et(fit_epoch)/cs.DAYS
    if '_' in desig:
        target = desig.split('_')[-2] + ' ' + desig.split('_')[-1]
    else:
        target = desig
    epoch  = 2451545.0 + et0 # TDB, _not_ UTC!
    query = Horizons(target, location='@10', epochs=epoch)
    vec = query.vectors(refplane='earth')
    xH = np.array([vec[key][0] for key in ['x', 'y', 'z', 'vx', 'vy', 'vz']])
    return xH



def InitOrbDet(Data, i1, i2, i3):
    tau1 = Data['et'][i1]-Data['et'][i2]
    tau3 = Data['et'][i3]-Data['et'][i2]
    Ri_ = Data['Rs'][i1,:], Data['Rs'][i2,:], Data['Rs'][i3,:]
    e1_ = spice.radrec(1., Data['ra'][i1], Data['de'][i1])
    e2_ = spice.radrec(1., Data['ra'][i2], Data['de'][i2])
    e3_ = spice.radrec(1., Data['ra'][i3], Data['de'][i3])
    ei_ = e1_, e2_, e3_
    r2i = 3.0
    kmax = 50
    tol = 1e-6
    r2_, v2_, k = AnglesOnlyIOD(tau1, tau3, Ri_, ei_, mu0, r2i, kmax, tol)
    if k < kmax-1:
       print('Preliminary orbit determination converged in %i iterations' % k)
       exit_code = 0
    else:
       print('WARNING: Preliminary orbit determination did not converge in %i iterations' % kmax)
       exit_code = 1
    return r2_, v2_, exit_code



def AnglesOnlyIOD(tau1, tau3, Ri_, Li_, mu_s, r2i, kmax, tol):
    ###
    def f_and_g(r0_, v0_, dt, mu):
        r0 = np.linalg.norm(r0_)
        v0 = np.linalg.norm(v0_)
        sigma = np.dot(r0_, v0_)/np.sqrt(mu)
        alpha = 2./r0 - v0**2/mu
        # initial guess for universal anomaly x
        x = np.sqrt(mu) * np.abs(alpha) * dt
        for k in range(50):
           z = alpha * x * x
           # evaluate Stumpff functions C, S
           if z > 0:
               C = (1. - np.cos(np.sqrt(z)))/z
               S = (np.sqrt(z) - np.sin(np.sqrt(z)))/np.sqrt(z)**3
           elif z < 0:
               C = (np.cosh(np.sqrt(-z)) - 1.)/(-z)
               S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/np.sqrt(-z)**3
           else:
               C = 1./2.
               S = 1./6.
           # Newton iteration for x  
           F    = x * x * x * S + sigma * x * x * C    + r0 * x * (1-z*S) - np.sqrt(mu)*dt
           dFdx = x * x * C     + sigma * x * (1.-z*S) + r0 * (1-z*C)
           dx   = - F / dFdx
           x = x + dx
           if abs(dx) < 1e-10:
               break
        f = 1  - x * x * C / r0
        g = dt - x * x * x * S / np.sqrt(mu)
        return f, g
    ###
    R1_, R2_, R3_ = Ri_
    L1_, L2_, L3_ = Li_
    r2 = r2i
    u2 = mu_s/r2**3
    f1 = 1 - u2*tau1**2/2
    f3 = 1 - u2*tau3**2/2
    g1 = tau1 - u2*tau1**3/6
    g3 = tau3 - u2*tau3**3/6
    delta = 1
    I_ = np.array([1,0,0])
    J_ = np.array([0,1,0])
    for k in range(kmax):
        r20 = r2
        A = np.array([ np.concatenate([f1*np.cross(I_,L1_), g1*np.cross(I_,L1_)]),
                       np.concatenate([f1*np.cross(J_,L1_), g1*np.cross(J_,L1_)]),
                       np.concatenate([   np.cross(I_,L2_), np.zeros(3)        ]),
                       np.concatenate([   np.cross(J_,L2_), np.zeros(3)        ]),
                       np.concatenate([f3*np.cross(I_,L3_), g3*np.cross(I_,L3_)]),
                       np.concatenate([f3*np.cross(J_,L3_), g3*np.cross(J_,L3_)]) ])
        b = np.array([ np.cross(L1_,R1_)[0], np.cross(L1_,R1_)[1],
                       np.cross(L2_,R2_)[0], np.cross(L2_,R2_)[1],
                       np.cross(L3_,R3_)[0], np.cross(L3_,R3_)[1] ])
        x = np.linalg.solve(A, b)
        r2_ = x[:3]
        v2_ = x[3:]
        r2 = np.linalg.norm(r2_)
        u2 = mu_s/r2**3
        p2 = np.dot(r2_,v2_)/r2**2
        q2 = np.dot(v2_,v2_)/r2**2 - u2
        f1, g1 = f_and_g(r2_, v2_, tau1, mu_s)
        f3, g3 = f_and_g(r2_, v2_, tau3, mu_s)
        delta = (r2-r20)/r20
        #print(('%4i'+6*'%14.6e') % (k, r2, f1, f3, g1, g3, delta))
        if abs(delta) < tol:
            break
    return r2_, v2_, k
 
