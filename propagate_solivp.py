import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as spice

import constants as cs


MU_S = 132712440041.279419 # km^3/s^2


### parameters for propagation with SciPy ode solver:
# gravitational parameters of planets (DE440, km^3/s^2)
MU_P = [22031.868551, 324858.592000, 398600.435507, 42828.375816, \
        126712764.100000, 37940584.841800, 5794556.400000, \
        6836527.100580, 975.500000, 4902.800118]
# gravitational parameter of asteroids (DE440, km^3/s^2)
MU_A = [62.628889, 13.665878, 1.920571, 17.288233, 0.646878, 1.139872,\
        5.625148, 2.023021, 1.589658, 0.797801, 2.683036, 0.938106, \
        2.168232, 1.189808, 3.894483, 2.830410]
# names of planets in SPICE Kernel
TAG_P = ['1', '2', '399', '4', '5', '6', '7', '8', '9', '301']
# names of asteroids in SPICE Kernel
TAG_A = ['2000001', '2000002', '2000003', '2000004', '2000006', '2000007', \
       '2000010', '2000015', '2000016', '2000029', '2000052', '2000065', \
       '2000087', '2000088', '2000511', '2000704']

# rescale constants to use units of au, days:
MU_S = MU_S * cs.DAYS**2 / cs.AU**3
MU_P = [ MU_P_i * (cs.DAYS**2 / cs.AU**3) for MU_P_i in MU_P ]
MU_A = [ MU_A_i * (cs.DAYS**2 / cs.AU**3) for MU_A_i in MU_A ]
CC = cs.CC * cs.DAYS / cs.AU


def PropagateWithSolIVP(x, et0, et, Rs, scipy_params, n_tau_iter=2):
    method, nongrav_coeffs, nongrav_accel = scipy_params
    rtol = 1e-9
    atol = 1e-11
    n = len(x)
    m = len(et)
    n_p = n-6  # number of parameters beyond initial state vector
    r0_, v0_, parms_ = x[0:3], x[3:6], x[6:]
    # form initial conditions for ODE integrator
    y0 = np.concatenate([r0_, v0_, np.eye(6).flatten(), np.zeros((6, n_p)).flatten()])
    # forward integration from et0 to et[-1]
    tspan_f = [et0, et[-1]]
    sol_f = solve_ivp(Derivs, tspan_f, y0, method=method, args=(parms_, nongrav_coeffs, nongrav_accel),
            rtol=rtol, atol=atol, dense_output=True)
    # backward integration from et0 to et[0]
    tspan_b = [et0, et[0]]
    sol_b = solve_ivp(Derivs, tspan_b, y0, method=method, args=(parms_, nongrav_coeffs, nongrav_accel),
            rtol=rtol, atol=atol, dense_output=True)
    ii_f = np.where(et >  et0)[0]
    ii_b = np.where(et <= et0)[0]
    ### iteration needed to account for light travel time:
    tau = np.zeros(m)
    for j in range(n_tau_iter):
        teval_b = et[ii_b]-tau[ii_b]
        teval_f = et[ii_f]-tau[ii_f]
        sol = np.array([sol_b.sol(tb).T for tb in teval_b] + [sol_f.sol(tf).T for tf in teval_f])
        tau = np.array([np.linalg.norm(sol[i,:3]-Rs[i,:])/CC for i in range(m)])
    # prepare output
    y = np.reshape(sol[:,  :6], (m, 6))
    P = np.reshape(sol[:,6:42], (m, 6, 6))
    S = np.reshape(sol[:,42: ], (m, 6, n_p))
    return y, P, S



def Derivs(t, y, parms_, nongrav_coeffs, nongrav_accel):
    r_ = y[0:3]
    v_ = y[3:6]
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    ## >>> call non-gravitational acceleration function:
    n_p = len(parms_)
    if n_p == 0:
       aNG_, dadrNG, dadvNG, dadpNG = np.zeros(3), np.zeros((3,3)), np.zeros((3,3)), []
    else:
       aNG_, dadrNG, dadvNG, dadpNG = nongrav_accel(r_, v_, parms_, nongrav_coeffs)
    ## <<<
    ### acceleration
    a_ = -MU_S*r_/r**3
    # add planets
    for i, TAG in enumerate(TAG_P):
        s_ = spice.spkpos(TAG, t*cs.DAYS, 'J2000', 'NONE', '10')[0]/cs.AU
        a_ += - MU_P[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # add asteroids
    for i, TAG in enumerate(TAG_A):
        s_ = spice.spkpos(TAG, t*cs.DAYS, 'J2000', 'NONE', '10')[0]/cs.AU
        a_ += - MU_A[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # add GR correction
    a_ += (MU_S/CC**2/r**3)*( (4*MU_S/r - v**2)*r_ + 4*np.dot(r_,v_)*v_ )
    # add non-gravitational term
    a_ = a_ + aNG_
    ### variational equations - note: asteroids, relativity omitted (GR would contribute to dadv)
    # 1. state transition matrix variation:
    PHI = np.reshape(y[6:42], (6, 6))
    ## dadr
    dadr = -MU_S/r**3*( np.eye(3) - 3*np.outer(r_, r_)/r**2)
    # add planets
    for i, TAG in enumerate(TAG_P):
        s_ = spice.spkpos(TAG, t*cs.DAYS, 'J2000', 'NONE', '10')[0]/cs.AU
        dadr += - MU_P[i]/np.linalg.norm(r_-s_)**3 \
          * ( np.eye(3) - 3*np.outer(r_-s_, r_-s_)/np.linalg.norm(r_-s_)**2 )
    # add NG term (set to zero above if not to be modeled)
    dadr += dadrNG
    ## dadv: the only contribution considered is from NG, if present (already zero otherwise)
    dadv = dadvNG
    ## dadp: the only contribution is from NG, if present
    dadp = dadpNG
    # variational equation matrix
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [dadr, dadv]])
    dPHIdt = A @ PHI
    # 2. sensitivity matrix variation:
    if n_p > 0:
        S = np.reshape(y[42:], (6, n_p))
        dSdt = A @ S + np.r_[np.zeros((3, n_p)), dadp]
    else:
        dSdt = np.array([])
    ### full vector with derivatives 
    dydt = np.r_[v_, a_, dPHIdt.flatten(), dSdt.flatten()]
    return dydt



def NonGravAccel(r_, v_, parms_, nongrav_coeffs):
    def X(w_):
        wx, wy, wz = w_
        return np.array([[0., -wz, +wy],
                         [+wz, 0., -wx],
                         [-wy, +wx, 0.]])

    alpha, r0, m, n, k = nongrav_coeffs

    r = np.linalg.norm(r_)
    h_ = np.cross(r_, v_)
    h = np.linalg.norm(h_)

    # --- RSW frame ---
    uR_ = r_ / r
    uW_ = h_ / h
    uS_ = np.cross(uW_, uR_)

    # --- derivatives of unit vectors ---
    I = np.eye(3)
    Xv = X(v_)
    Xr = X(r_)
    XuR = X(uR_)
    XuW = X(uW_)

    PR = I - np.outer(uR_, uR_)
    PW = I - np.outer(uW_, uW_)

    duRdr = PR / r
    duRdv = np.zeros((3, 3))
    duWdr = (PW / h) @ (-Xv)
    duWdv = (PW / h) @ (+Xr)
    duSdr = XuW @ duRdr - XuR @ duWdr
    duSdv = -XuR @ duWdv

    dudr = [duRdr, duSdr, duWdr]
    dudv = [duRdv, duSdv, duWdv]

    # --- Marsden g(r) ---
    rho = r / r0
    rho_n = rho**n
    g = alpha * rho**(-m) * (1. + rho_n)**(-k) * cs.NGASCALE
    dgdr = -g / r * (m + k * n / (1. + 1. / rho_n))

    # --- initialize ---
    n_p = len(parms_)
    aNG_ = np.zeros(3)
    dadpNG = np.zeros((3, n_p))
    dadrNG = np.zeros((3, 3))
    dadvNG = np.zeros((3, 3))

    if n_p < 4:

        n_eff = min(n_p, 3)
        for i in range(n_eff):
            Ai = parms_[i]
            ui = [uR_, uS_, uW_][i]
            aNG_       += g * Ai * ui
            dadpNG[:, i] = g * ui
            dadrNG     += g * Ai * dudr[i] + dgdr * Ai * np.outer(ui, uR_)
            dadvNG     += g * Ai * dudv[i]

    elif n_p == 4:

        A1, A2, A3, DT = parms_
        # get position and velocity at time (t-DT) via Keplerian propagation
        sv1 = spice.prop2b(MU_S, np.r_[r_, v_], -DT)
        r1_ = sv1[0:3]
        v1_ = sv1[3:6]
        r1 = np.linalg.norm(r1_)
        # g(r1), where r1 = r(t-DT)
        rho1 = r1 / r0
        rho1_n = rho1 ** n
        g1 = alpha * rho1**(-m) * (1. + rho1_n)**(-k) * cs.NGASCALE
        # dg/dr1, dr1/d(DT) -> dg(r1)/d(DT)
        dgdr1 = - g1 / r1 * ( m + k * n / (1. + 1./rho1_n) )
        # minus sign in following line comes from d(t-DT)/d(DT) = -1
        dr1dDT = - np.dot(r1_, v1_) / r1
        dg1dDT = dgdr1 * dr1dDT 
        # note: only partial consistency in dadrNG, dadvNG, dadpNG
        dadpNG[:, 3] = 0.0
        for i in range(3):
            Ai = parms_[i]
            ui = [uR_, uS_, uW_][i]
            aNG_   += g1 * Ai * ui
            dadrNG += g1 * Ai * dudr[i] + dgdr1 * Ai * np.outer(ui, uR_)
            dadvNG += g1 * Ai * dudv[i]
            dadpNG[:, i] = g1 * ui
            dadpNG[:, 3] += dg1dDT * Ai * ui

    return aNG_, dadrNG, dadvNG, dadpNG


