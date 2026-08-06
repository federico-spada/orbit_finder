import numpy as np
import spiceypy as spice
import constants as cs
from collections import Counter
from dataclasses import dataclass


@dataclass
class OrbitFit:
    et0: float
    x: np.ndarray
    Cx: np.ndarray
    z: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    flags: np.ndarray
    chi2: np.ndarray
    RMS: float

    @property
    def chisquare(self):
        return np.sum(self.chi2[self.flags])

    @property
    def dof(self):
        return 2.*len(self.y[self.flags]) - len(self.x)

    @property
    def reduced_chisquare(self):
        return self.chisquare / self.dof



def DifferentialCorrection(
    Data, fit_epoch, x0, propagator, prop_args,
    max_iter=5,
    chi2_rec=7.0,
    chi2_rej=8.0,
    RMS_tol=1e-6,
    dxw_tol=1e-7,
    dx2_tol=1e-12,
    weights=None,
    verbose=True,
):

    # initializations
    n = len(x0)
    m = Data.numObs
    et0 = spice.str2et(fit_epoch) / cs.DAYS
    x = x0.copy()
    Cx = np.zeros((n, n))
    flags = np.ones(m, dtype=bool)
    chi2 = np.zeros(m)
    m_use = m
    m_rec = 0
    m_rej = 0 
    RMSh = []
    ### weights are meant to represent down-weights:
    # w = np.inf: weight = 1 AND never reject
    # w = 1: fully trusted obs.
    # w < 1: less trusted obs.:
    # applied to C, W inside main loop
    if weights is None:
       weights = np.ones(m)
    # observational covariance matrix 
    C = np.zeros((m, 2, 2))
    C[:, 0, 0] = Data.rmsRAs**2
    C[:, 1, 1] = Data.rmsDec**2
    C[:, 0, 1] = Data.rmsCorr * Data.rmsRAs * Data.rmsDec
    C[:, 1, 0] = C[:, 0, 1]
    # observational weight matrix: Wi = Ci^-1
    W = np.linalg.inv(C)
    # further down-weight over-observed nights from same station
    stn = Data.stn
    night = np.floor(Data.et+0.5).astype(int)
    groups = Counter(zip(stn, night))
    for i in range(m):
        ngrp = groups[(stn[i], night[i])]
        if ngrp >= 4:
            weights[i] *= 4.0 / ngrp
    # variables for ResidualsAndPartials
    et = Data.et
    Rs = Data.Rs
    cos_ra = np.cos(Data.ra)
    sin_ra = np.sin(Data.ra)
    cos_de = np.cos(Data.dec)
    sin_de = np.sin(Data.dec)
    A = np.column_stack((-sin_ra, cos_ra, np.zeros_like(cos_ra)))
    D = np.column_stack((-sin_de * cos_ra,-sin_de * sin_ra, cos_de))    
    L = np.column_stack(( cos_de * cos_ra, cos_de * sin_ra, sin_de))

    # residuals from initial x
    z, B, y = ResidualsAndPartials(A, D, L, propagator, x, et0, et, Rs, prop_args)

    if verbose:
        print('Differential correction begins.')
        print('#iter.     RMS        chi2v       ||dx||_M     ',
        '||dx||_2    used    recv   rejc')

    for k in range(max_iter):

        # special case: never reject data with weight = np.inf
        force_keep = np.isinf(weights)
        flags[force_keep] = True

        # special case: always reject data with weight = 0.0
        force_drop = (weights == 0.0) 
        flags[force_drop] = False

        # assemble normal equations
        BTWB = np.zeros((n, n))
        BTWz = np.zeros(n)
        for i in range(m):
            if not flags[i]:
                continue
            zi = z[i]
            Bi = B[i]
            if (not force_keep[i]) and (not force_drop[i]):
                Wi = W[i] * weights[i]
            else:
                Wi = W[i] 
            # accumulate normal equations
            BTWB += Bi.T @ Wi @ Bi
            BTWz += Bi.T @ Wi @ zi

        # solve normal equations
        dx = np.linalg.solve(BTWB, BTWz)
        Cx = np.linalg.inv(BTWB)

        # apply correction
        x = x - dx

        # correction norm
        dx_norm2 = np.linalg.norm(dx)
        dx_normW = np.sqrt(dx.T @ BTWB @ dx)

        # post-fit residuals (from updated x)
        z, B, y = ResidualsAndPartials(A, D, L, propagator, x, et0, et, Rs, prop_args)

        # calculate chi-square (individual observations and total)
        chisqrd = 0.0
        for i in range(m):
            zi = z[i]
            Bi = B[i]
            if (not force_keep[i]) and (not force_drop[i]):
                Ci = C[i] / weights[i]
            else:
                Ci = C[i]
            if flags[i]:
                # observation i was used in the fit:
                Gi = Ci - (Bi @ Cx @ Bi.T)
            else:
                # observation i was not used in the fit:
                Gi = Ci + (Bi @ Cx @ Bi.T)
            chi2[i] = zi.T @ np.linalg.solve(Gi, zi)

        ## handling outliers 
        # recover observations 
        recover = (chi2 < chi2_rec) & (~flags) & (~force_drop)
        m_rec = recover.sum()
        flags[recover] = True
        # reject observations
        # variable rejection threshold
        chi2_rej_1 = chi2_rej + 400. * 1.2 ** -m_use
        chi2_rej_2 = 0.25 * np.percentile(chi2[flags], 95)
        chi2_rej_adj = max( chi2_rej_1, chi2_rej_2 )
        # mark rejection candidates
        i_mrk = np.where((flags) & (chi2 > chi2_rej_adj))[0]
        # cap number of rejected observations per iteration
        reject = np.zeros(m, dtype=bool)
        if i_mrk.size:
            idx_sorted = i_mrk[np.argsort(chi2[i_mrk])[::-1]]
            n_rej_iter = min(i_mrk.size, int(0.05 * m_use))
            reject[idx_sorted[:n_rej_iter]] = True
        m_rej = reject.sum()
        flags[reject] = False
        # update m_use 
        m_use = flags.sum()
        ## <<<

        # convergence diagnostics
        #RMS = np.sqrt(np.sum(z[flag]**2) / (2.0 * m_use))
        RMS = np.sqrt( np.mean(z[flags]**2) )
        chisqrd = np.sum(chi2[flags]) / (2.0 * m_use - n)
        RMSh.append(RMS)
        RMSm = np.mean(RMSh[-5:]) if len(RMSh) >= 5 else np.inf
        # screen output
        if verbose:
            print('%4i   %12.6e %12.6e %12.6e %12.6e %6i %6i %6i' %
                  (k, RMS, chisqrd, dx_normW, dx_norm2, m_use, m_rec, m_rej))

        # test convergence
        if k < 5:
            convergence = (dx_normW < dxw_tol) or (dx_norm2 < dx2_tol)
        else:
            convergence = (
                (dx_normW < dxw_tol) or
                (dx_norm2 < dx2_tol) or
                (abs(RMS/RMSm - 1.0) < RMS_tol)
            )
        # stop if convergence criterion is met
        if convergence:
            break

    if verbose:
        print('End of differential correction.')
    return OrbitFit(
        et0=et0,
        x=x,
        Cx=Cx,
        z=z,
        y=y,
        weights=weights,
        flags=flags,
        chi2=chi2,
        RMS=RMS
    )


def ResidualsAndPartials(A, D, L, propagator, x, et0, et, Rs, prop_args):

    n, m = len(x), len(et)

    y, P, S = propagator(x, et0, et, Rs, prop_args)

    B = np.zeros((m, 2, n))
    z = np.zeros((m, 2))

    for i in range(m):

        rho_vec = y[i, :3] - Rs[i]
        rho = np.linalg.norm(rho_vec)

        dL = (L[i] - rho_vec / rho)

        # residuals (RA*, Dec), in arc sec!
        z[i, 0] = np.dot(dL, A[i]) * cs.ARCSEC
        z[i, 1] = np.dot(dL, D[i]) * cs.ARCSEC

        # [ state transition matrix , sensitivity matrix ]
        # [ ∂y/∂y0 | ∂y/dp ] -> (6 , n)
        PS = np.concatenate([P[i], S[i]], axis=1)
        # observation partials
        # [ ∂v/∂y ] -> (2 , 6)
        rows = np.zeros((2, 6))
        rows[0, :3] = A[i]
        rows[1, :3] = D[i]

        # partials of the residuals wrt x, in arc sec!  
        # [ ∂v/∂x ] -> (2 , n) = (2 , 6) @ (6 , n)
        B[i] = - (rows / rho @ PS) * cs.ARCSEC


    return z, B, y


