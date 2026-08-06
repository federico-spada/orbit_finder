import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import spiceypy as spice

import constants as cs

mu0 = 132712440041.279419 * cs.DAYS**2 / cs.AU**3


def SummaryPlot(desig, Data, Fit, scaled=True, nbins=20, ylim=(-5, 5), fname='summary_plot.pdf'):
    m = Data.numObs
    if scaled:
       res_ra, res_de, unit = Fit.z[:,0] / Data.rmsRAs, Fit.z[:,1] / Data.rmsDec, '($\\sigma$)' 
       mask = np.isfinite(Fit.weights)
       res_ra[mask] /= np.sqrt(Fit.weights[mask])
       res_de[mask] /= np.sqrt(Fit.weights[mask])
    else:
       res_ra, res_de, unit = Fit.z[:,0], Fit.z[:,1], '(\")'
    dh = np.linalg.norm(Fit.y[:,:3], axis=1)
    rE, _ = spice.spkpos('399', Data.et*cs.DAYS, 'J2000', 'NONE', '10')
    dg = np.linalg.norm(Fit.y[:,:3]-rE/cs.AU, axis=1)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ### make plot
    fig, axes = plt.subplot_mosaic('AAAX;BBBY;CCCZ', figsize=(7, 7), constrained_layout=True)
    # plot heliocentric and geocentric distance
    axes['A'].plot(Data.utcDate, dh, '.', label='heliocentric')
    axes['A'].plot(Data.utcDate, dg, 's', ms=3, label='geocentric')
    axes['A'].xaxis.set_ticklabels([])
    axes['A'].set_ylabel('Distance (AU)')
    axes['A'].set_ybound(lower=0.)
    axes['A'].legend()  
    # plot residuals in RA 
    axes['B'].axhspan(-1, 1, color='palegreen', alpha=0.5)
    axes['B'].plot(Data.utcDate[Fit.flags], res_ra[Fit.flags], '.', color='#00356B', label='included')
    axes['B'].plot(Data.utcDate[~Fit.flags], res_ra[~Fit.flags], 'x', ms=4, color='C7', label='excluded')
    axes['B'].axhline(color='orangered', lw=0.7)
    axes['B'].xaxis.set_ticklabels([])
    axes['B'].set_ylabel('RA$^*$ Residuals '+unit)
    axes['B'].set_ylim(ylim)
    #axes['B'].legend()
    axes['B'].grid()
    # plot residuals in DE
    axes['C'].axhspan(-1, 1, color='palegreen', alpha=0.5)
    axes['C'].plot(Data.utcDate[Fit.flags], res_de[Fit.flags], '.', color='#00356B', label='Decl.')
    axes['C'].plot(Data.utcDate[~Fit.flags], res_de[~Fit.flags], 'x', ms=4, color='C7')
    axes['C'].axhline(color='orangered', lw=0.7)
    axes['C'].set_ylabel('Dec Residuals '+unit)
    axes['C'].set_xlabel('Date (UTC)')
    axes['C'].xaxis.set_major_locator(locator)
    axes['C'].xaxis.set_major_formatter(formatter)
    axes['C'].set_ylim(ylim)
    axes['C'].grid()
    axes['X'].axis('off')
    axes['X'].text(-0.1, 0.9, 'Fit Epoch:')
    axes['X'].text(-0.1, 0.8, spice.et2utc(Fit.et0 * cs.DAYS,'C', 0))
    axes['X'].text(-0.1, 0.65, 'Object:'+desig)
    axes['X'].text(-0.1, 0.55, 'RMS = '+str('%8.4f' % Fit.RMS)+' arc sec')
    axes['X'].text(-0.1, 0.45, 'chi-square = '+str('%8.4f' % Fit.reduced_chisquare))
    # add histograms
    axes['Y'].hist(res_ra[Fit.flags], bins=nbins, orientation='horizontal', color='#00356B')
    axes['Y'].axhline(color='orangered', lw=0.7)
    axes['Y'].set_ylim(ylim)
    axes['Y'].grid()
    axes['Z'].hist(res_de[Fit.flags], bins=nbins, orientation='horizontal', color='#00356B')
    axes['Z'].axhline(color='orangered', lw=0.7)
    axes['Z'].set_xlabel('Count')
    axes['Z'].set_ylim(ylim)
    axes['Z'].grid()
    plt.savefig(fname)
    plt.close()


def SummaryText(desig, Data, Fit, fname='summary_fit.txt'):
    n = len(Fit.x) 
    xs1 = ['x ', 'y ', 'z ', 'vx', 'vy', 'vz', 'A1', 'A2', 'A3', 'DT']
    xs2 = np.r_[np.repeat('(au)', 3), np.repeat('(au/d)', 3), np.repeat('(10^-8 au/d^2)', 3),
    ['(days)']]
    ee1 = ['e', 'a', 'q', 'I', 'Ω', 'ω', 'M']
    ee2 = [' ', '(au)', '(au)', '(deg)', '(deg)', '(deg)', '(deg)']
    U = spice.pxform( 'J2000', 'ECLIPJ2000', Fit.et0 * cs.DAYS )
    r_, v_ = U @ Fit.x[0:3],   U @ Fit.x[3:6]
    q, e, I, W, w, M, _, _, _, a, _ = spice.oscltx(np.r_[r_, v_], Fit.et0, mu0)
    ee = np.array([e, a, q, np.degrees(I), np.degrees(W), np.degrees(w), np.degrees(M)])
    A = JacobianTBP(r_, v_, mu0)[1]
    CovOE = A @ Fit.Cx[:6,:6] @ A.T
    ss_q = np.sqrt((1.-e)**2 * CovOE[0,0] + a**2 * CovOE[1,1] -2.*a*(1.-e) * CovOE[0,1])
    ss_a, ss_e, ss_I, ss_W, ss_w, ss_M = np.sqrt(np.diagonal(CovOE))
    sigma_ee = np.array([ss_e, ss_a, ss_q, ss_I, ss_W, ss_w, ss_M])
    sigma_ee[3:] *= 180./np.pi
    with open(fname, 'w') as f:
        f.write('orbit_finder - '+datetime.today().strftime('%Y-%m-%d %H:%M:%S')+'\n\n')
        if 'C_' in desig:
            f.write(desig.replace('C_','C/').replace('_',' ')+'\n')
        else:
            f.write('Object name: '+desig+'\n')
        f.write('\n')
        f.write('RMS = %8.4f (arc sec)\n' % Fit.RMS)
        f.write('χ^2 = %8.4f\n' % Fit.reduced_chisquare)
        f.write('# residuals = %5i \n' % (2*sum(Fit.flags)) )
        f.write('\n')
        f.write('# observations included = %5i \n' % sum(Fit.flags))
        f.write('# observations rejected = %5i \n' % sum(~Fit.flags))
        f.write('# total                 = %5i \n' % Data.numObs)
        f.write('\n')
        f.write('First epoch: '+spice.et2utc(Data.et[ 0]*cs.DAYS, 'C', 2)+'\n')
        f.write('Last  epoch: '+spice.et2utc(Data.et[-1]*cs.DAYS, 'C', 2)+'\n')
        f.write('\n')        
        f.write('Fit Epoch  : '+spice.et2utc(Fit.et0*cs.DAYS, 'C', 2)+'\n')
        f.write('\n')
        f.write('State vector (J2000 heliocentric frame)\n')
        for i in range(n):
            x_i = Fit.x[i]
            sx_i = np.sqrt(Fit.Cx[i,i])
            if len(Fit.x) <=10:
                f.write('%s = %15.9f ± %13.6e %s\n' % (xs1[i], x_i, sx_i, xs2[i]))
            else:
                f.write('x%02i = %15.9f ± %13.6e\n' % (i, x_i, sx_i))
        f.write('\n')
        f.write('Orbital elements (ECLIPJ2000 heliocentric frame)\n')          
        for i in range(len(ee)):
            f.write('%s = %15.9f ± %13.6e %s\n' % (ee1[i], ee[i], sigma_ee[i], ee2[i]))
        f.write('\n')


def JacobianTBP(r_, v_, mu):
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    # get osculating orbital elements:
    a = 1./(2./r - v**2/mu)
    h_ = np.cross(r_, v_)
    h = np.linalg.norm(h_)
    e_ = np.cross(v_, h_)/mu - r_/r
    e = np.linalg.norm(e_)
    W = np.arctan2(h_[0], -h_[1])
    n_ = np.array([ np.cos(W), np.sin(W), 0])
    b_ = np.cross(h_/h, n_)
    w = np.arctan2( np.dot(e_, b_), np.dot(e_, n_) )
    # for the I, W, w partials:
    W_ = h_/h
    P_ = e_/e
    Q_ = np.cross(W_, P_)
    X, Y, Xd, Yd = np.dot(r_,P_), np.dot(r_,Q_), np.dot(v_,P_), np.dot(v_,Q_)
    drdI_, dvdI_ = (X*np.sin(w)+Y*np.cos(w))*W_, (Xd*np.sin(w)+Yd*np.cos(w))*W_
    drdW_, dvdW_ = np.array([-r_[1], r_[0], 0.]), np.array([-v_[1], v_[0], 0.])
    # for the a, e, Mo partials:
    if e < 1.:
        n = np.sqrt(mu/a**3)
        j = np.sqrt(1.-e**2)
        L  = -a - Y**2/r/j**2
        M  = X*Y/r/j**2
        Ld =     Xd*(a/r)**2*( 2*(X/a) + e*(Y/a)**2/j**2 )
        Md =  (n/j)*(a/r)**2*( X**2/r - Y**2/a/j**2 )
        # Fundamental matrix: d(state)/d(elts)
        R = np.zeros((6,6))
        R[:,0] = np.r_[ r_/a, -v_/2/a ]           # ds/da 
        R[:,1] = np.r_[ L*P_+M*Q_, Ld*P_+Md*Q_ ]  # ds/de 
        R[:,2] = np.r_[ drdI_, dvdI_ ]            # ds/dI
        R[:,3] = np.r_[ drdW_, dvdW_ ]            # ds/dW
        R[:,4] = np.r_[ X*Q_-Y*P_, Xd*Q_-Yd*P_]   # ds/dw
        R[:,5] = np.r_[ v_/n, -n*(a/r)**3*r_ ]    # ds/dMo 
    else:
        n = np.sqrt(mu/(-a)**3)
        j = np.sqrt(e**2-1.)
        L  = -a + Y**2/r/j**2
        M  = -X*Y/r/j**2
        Ld =     Xd*(a/r)**2*( 2*(X/a) - e*(Y/a)**2/j**2 )
        Md =  (n/j)*(a/r)**2*(-X**2/r - Y**2/a/j**2 )
        # Fundamental matrix: d(state)/d(elts)
        R = np.zeros((6,6))
        R[:,0] = np.r_[ r_/a, -v_/2/a ]           # ds/da 
        R[:,1] = np.r_[ L*P_+M*Q_, Ld*P_+Md*Q_ ]  # ds/de 
        R[:,2] = np.r_[ drdI_, dvdI_ ]            # ds/dI
        R[:,3] = np.r_[ drdW_, dvdW_ ]            # ds/dW
        R[:,4] = np.r_[ X*Q_-Y*P_, Xd*Q_-Yd*P_]   # ds/dw
        R[:,5] = np.r_[ v_/n, +n*(a/r)**3*r_ ]    # ds/dMo 
    # Inverse of the fundamental matrix: d(elts)/d(state)
    RI = np.linalg.inv(R)
    return R, RI
