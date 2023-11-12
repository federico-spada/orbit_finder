import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from astroquery.jplhorizons import Horizons
import spiceypy as spice
from scipy.integrate import solve_ivp
from extensisq import SWAG


# constants 
Re = 6378.1366 # km
AU   = 1.495978707e8 # km
days = 86400 # s 
mu_s = 132712440041.279419 # km^3/s^2

# gravitational parameters of planets (DE440)
mu_p = np.array([22031.868551, 324858.592000, 398600.435507, 42828.375816, \
                 126712764.100000, 37940584.841800, 5794556.400000, \
                 6836527.100580, 975.500000, 4902.800118])
# gravitational parameter of asteroids (DE440)
mu_a = np.array([62.628889, 13.665878, 1.920571, 17.288233, 0.646878, 1.139872,\
                 5.625148, 2.023021, 1.589658, 0.797801, 2.683036, 0.938106, \
                 2.168232, 1.189808, 3.894483, 2.830410])
# names of planets in SPICE Kernel
tgP = ['1', '2', '399', '4', '5', '6', '7', '8', '9', '301']
# names of asteroids in SPICE Kernel
tgA = ['2000001', '2000002', '2000003', '2000004', '2000006', '2000007', \
       '2000010', '2000015', '2000016', '2000029', '2000052', '2000065', \
       '2000087', '2000088', '2000511', '2000704']

# scaled to the same units used by assist
mu_s = mu_s * days**2/AU**3
mu_p = mu_p * days**2/AU**3
mu_a = mu_a * days**2/AU**3
cc = spice.clight() * days/AU

# Veres et al. 2017
uncertainty = { '703': 1.0 , '691': 0.6 , '644': 0.6 ,
                '704': 1.0 , 'G96': 0.5 , 'F51': 0.2 , 'G45': 0.6 , '699': 0.8 ,
                'D29': 0.75, 'C51': 1.0 , 'E12': 0.75, '608': 0.6 , 'J75': 1.0 ,
                '645': 0.3 , '673': 0.3 , '689': 0.5 , '950': 0.5 , 'H01': 0.3 ,
                'J04': 0.4 , 'G83': 0.3 , 'K92': 0.4 , 'K93': 0.4 , 'Q63': 0.4 ,
                'Q64': 0.4 , 'V37': 0.4 , 'W84': 0.4 , 'W85': 0.4 , 'W86': 0.4 ,
                'W87': 0.4 , 'K91': 0.4 , 'E10': 0.4 , 'F65': 0.4 , 'Y28': 0.3 ,
                '568': 0.25, 'T09': 0.1 , 'T12': 0.1 , 'T14': 0.1 , '309': 0.3 ,
                '250': 0.05} # add HST


### LOAD DATA -----------------------------------------------------------------
def LoadDataMPC(obsstat_file,objdata_file):
    # load locations of observing stations
    obss = np.array([])
    lons = np.array([])
    rcos = np.array([])
    rsin = np.array([])
    with open(obsstat_file) as f:
        for l in f:
            if l[3:12] != '         ':
               obss = np.append(obss, l[0:3])
               lons = np.append(lons, float(l[3:13]))
               rcos = np.append(rcos, float(l[13:21])*Re)
               rsin = np.append(rsin, float(l[21:30])*Re)
    # load observations
    OC = np.array([])
    JD = np.array([])
    ra = np.array([])
    de = np.array([])
    RS = np.zeros(3)
    et = np.array([])
    s_ra = np.array([])
    s_de = np.array([])
    count = 0
    with open(objdata_file) as f:
        for l in f:
            count += 1
            # obs. station code 
            code = l[77:80]
            OC = np.append(OC, code)
            # time of observation
            year  = float(l[15:19])
            month = float(l[20:22])
            day   = float(l[23:32])
            # JD and fraction (UTC) 
            A = np.floor( 7*( year + np.floor((month+9)/12) )/4 )
            B = np.floor(275*month/9)
            jdi = 367*year - A + B + day +  1721013.5 # note: day includes fraction 
            JD = np.append(JD, jdi)
            # ephemerides time from spice (equiv. to TDB)
            eti = spice.str2et(str('JDUTC %17.9f' % jdi))
            et = np.append(et, eti)
            # observed RA
            hours   = float(l[32:34])
            minutes = float(l[35:37])
            seconds = float(l[38:44])
            rao = hours + minutes/60 + seconds/3600
            ra = np.append(ra, np.deg2rad(15*rao)) # rad
            # observed Decl
            degrees = float(l[44:47])
            minutes = float(l[48:50])
            seconds = float(l[51:56])
            deo = np.sign(degrees)*(abs(degrees) + minutes/60 + seconds/3600)
            de = np.append(de, np.deg2rad(deo)) # rad
            # heliocentric position of Earth to observation point vector
            rE, _ = spice.spkpos('399', eti, 'J2000', 'NONE', '10')
            # geocentric position of observing station
            if code == '250':
                # for HST, data is in the file 
                l1 = f.readline()
                X = float(l1[34:42].replace(' ',''))
                Y = float(l1[46:54].replace(' ','')) 
                Z = float(l1[58:66].replace(' ',''))
                R_ = np.array([X,Y,Z]) 
            else:
                # for other observatories, must be calculated from geodetic data
                io = np.where(obss == code)[0]
                R0_ = spice.cylrec(rcos[io], np.deg2rad(lons[io]), rsin[io])
                # transform from e.f. to s.f. coordinates
                U = spice.pxform( 'ITRF93', 'J2000', eti )
                R_ = U @ R0_
            RS = np.vstack((RS, R_+rE))
            # uncertainties in RA, Decl: assume 10 arc sec
            s_ra = np.append(s_ra, np.deg2rad(10/3600)) # rad
            s_de = np.append(s_de, np.deg2rad(10/3600)) # rad
    RS = np.delete(RS, 0, 0)
    # use information from dictionary of uncertainties:
    for key in uncertainty:
        ii = np.where(OC == key)[0]
        s_ra[ii] = np.deg2rad(uncertainty[key]/3600) # arc sec to rad 
        s_de[ii] = np.deg2rad(uncertainty[key]/3600) # arc sec to rad 
    # change units to AU, days 
    RS = RS / AU
    et = et / days
    return et, ra, de, s_ra, s_de, RS, JD, OC




### PRELIMINARY ORBIT DETERMINATION -------------------------------------------
def PreliminaryOrbitDetermination(et,ra,de,s_ra,s_de,RS,i1,i2,i3):
    taui = et[i1]-et[i2], et[i3]-et[i2], et[i3]-et[i1]
    Ri_ = RS[i1,:], RS[i2,:], RS[i3,:]
    e1_ = spice.radrec(1.,ra[i1],de[i1])
    e2_ = spice.radrec(1.,ra[i2],de[i2])
    e3_ = spice.radrec(1.,ra[i3],de[i3])
    ei_ = e1_, e2_, e3_
    r2i = 3.0
    kmax = 50
    tol = 1e-6
    r2_, v2_, k = AnglesOnlyPOD(taui,Ri_,ei_,mu_s,r2i,kmax,tol)
    if k < kmax-1:
       print('Preliminary orbit determination converged in %i iterations' % k)
       exit_code = 0
    else:
       print('WARNING: Preliminary orbit determination did not converge in %i iterations' % kmax)
       exit_code = 1
    return r2_, v2_, exit_code

# Reference: Bate, Mueller, White (1971), Section 5.8, page 271
def AnglesOnlyPOD(taui,Ri_,Li_,mu_s,r2i,kmax,tol):
    tau1, tau3, tau = taui
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
                       np.concatenate([   np.cross(I_,L2_), np.array([0,0,0])  ]),
                       np.concatenate([   np.cross(J_,L2_), np.array([0,0,0])  ]),
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
        f1 = 1 - u2*tau1**2/2 + u2*p2*tau1**3/2 + u2*(u2 - 15*p2**2 + 3*q2)*tau1**4/24 \
               + u2*p2*(7*p2**2 - u2 - 3*q2)*tau1**5/8
        g1 = tau1 - u2*tau1**3/6 + u2*p2*tau1**4/4 + u2*(u2 - 45*p2**2 + 9*q2)*tau1**5/120
        f3 = 1 - u2*tau3**2/2 + u2*p2*tau3**3/2 + u2*(u2 - 15*p2**2 + 3*q2)*tau3**4/24 \
               + u2*p2*(7*p2**2 - u2 - 3*q2)*tau3**5/8
        g3 = tau3 - u2*tau3**3/6 + u2*p2*tau3**4/4 + u2*(u2 - 45*p2**2 + 9*q2)*tau3**5/120
        delta = (r2-r20)/r20
        #print(('%4i'+6*'%14.6e') % (k, r2, f1, f3, g1, g3, delta))
        if abs(delta) < tol:
            break
    return r2_, v2_, k



### DIFFERENTIAL CORRECTION OF THE ORBIT --------------------------------------
# References: Farnocchia et al. (2015) (general method); 
#             Carpino et al. (2003) (for outliers rejection)
def DifferentialCorrection(et,ra,de,s_ra,s_de,RS,et0,x0,ng_acc,kmax):
    # parameters
    X2_rjb = 8.   # base rejection threshold
    X2_rec = 7.   # recovery threshold
    alpha  = 0.25 # fraction of max chi-square to use as increased rejection threshold 
    frac   = 0.05 # maximum fraction of epochs discarded in a single step
    m = len(et) # number of epochs
    n = len(x0) # number of fitting parameters
    # uncertainties and weighting matrix
    s = np.r_[s_ra, s_de]
    W = np.diag(1./s**2)
    # initializations 
    x = x0
    z, B = ResidualsAndPartials(et,ra,de,RS,et0,x,ng_acc)
    Cov = np.linalg.inv(B.T @ W @ B) 
    flag = np.repeat(True,m) # True for epochs included in the fit
    X2 = np.zeros(m)
    m_use = m  
    # begin differential correction iteration
    print('Differential correction begins.')
    print('#iter   red.chisq.   metric      ||dx/x||        #rec   #rej  #use frac')
    for k in range(kmax):
        ### least-squares fit here
        #flag = np.repeat(True,m) # DEBUGGING ONLY 
        # solve normal equations and apply corrections to x:
        mask = np.r_[flag, flag]
        B1 = B[mask,:]
        W1 = W[mask,:][:,mask]
        C1 = B1.T @ W1 @ B1
        G1 = np.linalg.inv(C1)
        z1 = z[mask]
        dx = G1 @ B1.T @ W1 @ z1
        Cov = G1
        # update parameters vector:
        x = x + dx
        # get residuals and design matrix for updated x
        z, B = ResidualsAndPartials(et,ra,de,RS,et0,x,ng_acc)
        # calculate post-fit residuals
        u = ( np.eye(2*m) - B @ Cov @ B.T @ W ) @ z
        ### handle epoch selection based on chi-square
        for i in range(m):
            ui = np.array([u[i], u[m+i]]) # post-fit res. at epoch i
            Bi = np.vstack((B[i,:], B[m+i,:])) # relevant rows of design mat.
            Gi = np.array([[s[i]**2, 0.], [0., s[m+i]**2]]) # cov. of the u_i
            if flag[i]:
                # expected covariance of u_i if epoch was included in the fit
                Gui = Gi - Bi @ Cov @ Bi.T
            else:
                # expected covariance of u_i if epoch not included in the fit
                Gui = Gi + Bi @ Cov @ Bi.T
            # chi-square at epoch i
            X2[i] = ui @ np.linalg.inv(Gui) @ ui.T
        # rejection threshold is adjusted to not discard too many points
        X2_rej = max( X2_rjb + 400./(1.2)**m_use, alpha*np.max(X2[flag]) )
        # recover epochs 
        i_rec = np.where((flag == False) & (X2 < X2_rec))[0]
        flag[i_rec] = True
        m_rec = len(i_rec)        
        # reject epochs
        i_mrk = np.where((flag == True) & (X2 > X2_rej))[0]
        m_rej = min(len(i_mrk), int(frac*m_use))
        X2_mrk = np.flip(np.sort(X2[i_mrk]))
        if len(i_mrk) > 0:
            kkk =[np.where(X2 == X2_m)[0][0] for X2_m in X2[i_mrk][:m_rej]] 
            flag[kkk] = False 
        # update total number of epochs to be used in the fit
        m_use = sum(flag)
        ### convergence metrics
        mask = np.r_[flag, flag]
        chi2 = z[mask] @ W[mask,:][:,mask] @ z[mask]
        chi2v = chi2/(2*m_use-n)
        metric = np.sqrt((dx @ (B[mask,:].T @ W[mask,:][:,mask] @ B[mask,:]) @ dx)/n)
        ### screen output (comment out or leave if appropriate)
        print('%4i   %12.6e %12.6e %12.6e %6i %6i %6i %4.2f' % (k, chi2v, metric, 
               np.linalg.norm(dx), m_rec, m_rej, m_use, m_use/m))
        ### stopping condition
        if metric < 0.5 or chi2v < 0.5 or np.linalg.norm(dx/x) < 5e-8:
            break
    return x, Cov, z, chi2, B, flag, u, X2

def ResidualsAndPartials(et,ra,de,RS,et0,x,ng_acc):
    rtol = 1e-8
    atol = 1e-11
    n = len(x)
    m = len(et)
    nng = n-6 
    ###
    r0_, v0_, A = x[0:3], x[3:6], x[6:]
    # form initial conditions for ODE integrator
    y0 = np.concatenate([r0_, v0_, np.eye(6).flatten(), np.zeros((6,nng)).flatten()])
    # forward integration from et0 to et[-1]
    tspan_f = [et0, et[-1]]
    sol_f = solve_ivp(derivs,tspan_f,y0,method=SWAG,args=(A,ng_acc,),rtol=rtol,atol=atol,dense_output=True)
    # backward integration from et0 to et[0]
    tspan_b = [et0, et[0]] 
    sol_b = solve_ivp(derivs,tspan_b,y0,method=SWAG,args=(A,ng_acc,),rtol=rtol,atol=atol,dense_output=True)
    ii_f = np.where(et >  et0)[0]
    ii_b = np.where(et <= et0)[0]
    ### iteration needed to account for light travel time:
    tau = np.zeros(m)
    for j in range(2):
        teval_b = et[ii_b]-tau[ii_b]
        teval_f = et[ii_f]-tau[ii_f]
        yy = np.r_[sol_b.sol(teval_b).T, sol_f.sol(teval_f).T]
        for i in range(m):
            r_ = yy[i,:3]
            R_ = RS[i,:]
            # note speed of light in vacuum, cc, is in AU/days 
            tau[i] = np.linalg.norm(r_-R_)/cc
    ###
    # initialize residuals and their partials matrix for current x
    B = np.zeros((2*m,n))
    z = np.zeros(2*m)   
    # fill in B and z (Escobal 1976 formulae)
    for i in range(m):
        r_ = yy[i,:3]
        R_ = RS[i,:]
        P  = np.reshape(yy[i,6:42],(6,6))
        S  = np.reshape(yy[i,42:],(6,nng))
        rho_ = r_ - R_
        rho = np.linalg.norm(rho_)
        cos_ra, sin_ra = np.cos(ra[i]), np.sin(ra[i])
        cos_de, sin_de = np.cos(de[i]), np.sin(de[i])
        Lc_ = rho_/rho
        A_  = np.array([-sin_ra, cos_ra, 0.])
        D_  = np.array([-sin_de*cos_ra, -sin_de*sin_ra, cos_de ])
        Lo_ = np.array([ cos_de*cos_ra,  cos_de*sin_ra, sin_de ])
        dL_ = Lo_ - Lc_
        # fill partials matrix
        B[i  ,:] = np.r_[ A_, np.zeros(3) ]/rho @ np.block([P, S])
        B[m+i,:] = np.r_[ D_, np.zeros(3) ]/rho @ np.block([P, S])
        # fill residual vector
        z[i  ] = np.dot(dL_,A_) 
        z[m+i] = np.dot(dL_,D_)
    return z, B


def derivs(t,y,A,ng_acc):
    r_ = y[0:3]
    v_ = y[3:6]
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    # main term
    f_ = -mu_s*r_/r**3
    # perturbations
    p_ = 0
    # planets
    for i in range(len(mu_p)):
        s_, _ = spice.spkpos(tgP[i], t*days, 'J2000', 'NONE', '10')
        s_ = s_/AU 
        p_ = p_ - mu_p[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # asteroids
    for i in range(len(mu_a)):
        s_, _ = spice.spkpos(tgA[i], t*days, 'J2000', 'NONE', '10')
        s_ = s_/AU
        p_ = p_ - mu_a[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # GR correction
    p_ = p_ + (mu_s/cc**2/r**3)*( (4*mu_s/r - v**2)*r_ + 4*np.dot(r_,v_)*v_ )
    # non-gravitational term
    nng = len(A)
    g = (1.0/r)**2
    match ng_acc:
        case 'RTN':
            ur_ = r_/r
            un_ = np.cross(r_,v_)/np.linalg.norm(np.cross(r_,v_))
            ut_ = np.cross(un_,ur_)
            arad_ = g * (A[0] * ur_ + A[1] * ut_ + A[2] * un_)
            dadA  = g * np.c_[ur_, ut_, un_]
        case 'ACN':
            ua_ = v_/v
            un_ = np.cross(r_,v_)/np.linalg.norm(np.cross(r_,v_))
            uc_ = np.cross(un_,ua_)
            arad_ = g * (A[0] * ua_ + A[1] * uc_ + A[2] * un_)
            dadA  = g * np.c_[ua_, uc_, un_]
        case 'radial':
            arad_ = g * A[0] * r_/r
            dadA  = arad_/A[0]
        case 'tangential':
            arad_ = -g * A[0] * v_/v
            dadA  = arad_/A[0]
        case _:
            arad_ = np.zeros(3)
            dadA = np.array([])
    p_ = p_ + arad_
    # total acceleration
    a_ = f_ + p_
    ### variational equations
    PHI = np.reshape(y[6:42],(6,6))
    # main term
    F = -mu_s/r**3*( np.eye(3) - 3*np.outer(r_,r_)/r**2)
    # point mass perturbers (planets only!)
    P = np.zeros((3,3))
    for i in range(len(mu_p)):
        s_, _ = spice.spkpos(tgP[i], t*days, 'J2000', 'NONE', '10')
        s_ = s_/AU
        P = P - mu_p[i]/np.linalg.norm(r_-s_)**3 \
          * ( np.eye(3) - 3*np.outer(r_-s_,r_-s_)/np.linalg.norm(r_-s_)**2 )
    # asteroids, relativity, and radiation pressure are omitted in the variational eqns.
    G = F + P
    AP = np.block([[np.zeros((3,3)), np.eye(3)], [G, np.zeros((3,3))]]) @ PHI
    # sensitivity matrix
    if nng > 0:
        S = np.reshape(y[42:],(6,nng))
        AS = np.block([[np.zeros((3,3)), np.eye(3)], [G, np.zeros((3,3))]]) @ S \
           + np.concatenate((np.zeros((3,nng)),np.c_[dadA]))
    else:
        AS = np.array([])
    ### full vector with derivatives 
    dydt = np.concatenate([v_, a_, AP.flatten(), AS.flatten()])
    return dydt


### PLOT AND OUTPUT -----------------------------------------------------------
def PlotResiduals(et,res_ra,s_ra,res_de,s_de,flag,scaled=False):
    #plt.figure()
    plt.ion()
    plt.clf()
    not_flag = np.logical_not(flag)
    tlf = [datetime.fromisoformat(spice.et2utc(eti*days,'ISOC', 0)) for eti in et[not_flag]]
    tlt = [datetime.fromisoformat(spice.et2utc(eti*days,'ISOC', 0)) for eti in et[flag]]
    plt.subplot(211)
    if scaled:
        plt.plot(tlf,res_ra[not_flag]/s_ra[not_flag],'x',color='darkgray')
        plt.plot(tlt,res_ra[flag]/s_ra[flag],'.',color='#00356B',label='R.A.')
        plt.ylabel('R. A. Residuals ($\sigma$)') 
        #plt.ylim(min(res_ra[flag]/s_ra[flag]),max(res_ra[flag]/s_ra[flag]))
        plt.ylim(-12,12)
    else:
        plt.plot(tlf,3600*np.rad2deg(res_ra[not_flag]),'x',color='darkgray')
        plt.plot(tlt,3600*np.rad2deg(res_ra[flag]),'.',color='#00356B',label='R.A.')
        plt.ylabel('R. A. Residuals (arcsec)')
        plt.ylim(min(3600*np.rad2deg(res_ra[flag])),max(3600*np.rad2deg(res_ra[flag])))
    plt.grid()
    plt.gca().xaxis.set_ticklabels([])
    plt.subplot(212)
    if scaled:
        plt.plot(tlf,res_de[not_flag]/s_de[not_flag],'x',color='darkgray')
        plt.plot(tlt,res_de[flag]/s_de[flag],'.',color='#00356B',label='Decl.')
        plt.ylabel('Decl. Residuals ($\sigma$)')
        #plt.ylim(min(res_de[flag]/s_de[flag]),max(res_de[flag]/s_de[flag]))
        plt.ylim(-12,12)
    else:
        plt.plot(tlf,3600*np.rad2deg(res_de[not_flag]),'x',color='darkgray')
        plt.plot(tlt,3600*np.rad2deg(res_de[flag]),'.',color='#00356B',label='Decl.')
        plt.ylabel('Decl. Residuals (arcsec)')
        plt.ylim(min(3600*np.rad2deg(res_de[flag])),max(3600*np.rad2deg(res_de[flag])))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=15)
    plt.xlabel('Date (UTC)')
    plt.grid()
    plt.tight_layout()    

def ScreenOutput(et0,x,Cov):
    names = ['x0 ','y0 ','z0 ','vx0','vy0','vz0','A1 ','A2 ','A3 ']
    units = ['AU','AU','AU','AU/day','AU/day','AU/day',
             'AU/day^2','AU/day^2','AU/day^2']
    s_x = np.sqrt(np.diagonal(Cov)) 
    print(' Best-fitting parameters:')
    for k in range(len(x)):
        if k <= 5:
            print('    %s = %13.10f +/- %13.10f %s' % (names[k], x[k], s_x[k], units[k]))
        else:
            print('    %s = %13.6e +/- %13.6e %s' % (names[k], x[k], s_x[k], units[k]))
    U = spice.pxform( 'J2000', 'ECLIPJ2000', et0*days )
    r_, v_ = U @ x[0:3], U @ x[3:6]
    elts = spice.oscelt(np.r_[r_, v_], et0*days, mu_s)
    rp, ecc, inc, lnode, argp, M0, _, _ = elts
    print('Orbital elements at epoch ', spice.et2utc(et0*days, 'C', 2),':')
    print('a = %16.8f AU' % (rp/(1-ecc)))
    print('e = %16.8f   ' % ecc)
    print('i = %16.8f   ' % np.rad2deg(inc))
    print('Ω = %16.8f   ' % np.rad2deg(lnode))
    print('ω = %16.8f   ' % np.rad2deg(argp)) 
    print('M = %16.8f   ' % np.rad2deg(M0))




def RunFit(obj_name,i1,i2,i3,parm_,ng_acc='none',it_max=5):
    ### provide all Kernels via meta-Kernel
    spice.furnsh('spice.mkn')
    ### load data
    obsstat_file = 'mpc_obs.txt'
    objdata_file = obj_name+'.txt'
    et, ra, de, s_ra, s_de, RS, JD, OC = LoadDataMPC(obsstat_file,objdata_file)
    print('Object: ', obj_name)
    ### preliminary orbit determination  
    r2_, v2_, _ = PreliminaryOrbitDetermination(et,ra,de,s_ra,s_de,RS,i1,i2,i3)
    et0 = et[i2]
    ### to check final result
    epoch = float(spice.et2utc(et0*days,'J', 10)[3:])
    q = Horizons(obj_name, location='@sun', epochs=epoch)
    vec = q.vectors(refplane='earth')
    pos0_ = np.array([vec['x'][0], vec['y'][0], vec['z'][0]])
    vel0_ = np.array([vec['vx'][0], vec['vy'][0], vec['vz'][0]])
    ### differential correction of the orbit
    # initialize first guess
    x0 = np.r_[r2_,v2_,parm_]
    # run differential correction
    x, Cov, z, chi2, B, flag, u, X2 = DifferentialCorrection(et,ra,de,s_ra,s_de,RS,et0,x0,ng_acc,it_max)
    # output results and check
    #print('Converged solution:')
    #print((6*'%16.8e') % (x[0], x[1], x[2], x[3], x[4], x[5]))
    #print('JPL Horizons values for comparison:')
    #print((6*'%16.8e') % (pos0_[0], pos0_[1], pos0_[2], vel0_[0], vel0_[1], vel0_[2]))
    print('Differences with JPL Horizons values:')
    print((6*'%16.8e') % (pos0_[0]-x[0],pos0_[1]-x[1],pos0_[2]-x[2],vel0_[0]-x[3],vel0_[1]-x[4],vel0_[2]-x[5]))
    ### screen output
    ScreenOutput(et0,x,Cov)
    ### plot final residuals
    res_ra, res_de = z[:len(et)], z[len(et):]
    PlotResiduals(et,res_ra,s_ra,res_de,s_de,flag,scaled=True)
    ### release the spice Kernels 
    spice.kclear()


### MAIN ----------------------------------------------------------------------
if __name__ == "__main__":

    print('Oumuamua, no non-grav. acc.')
    RunFit('1I',5,15,30,np.array([]))
    print('Oumuamua, with non-grav. acc. (radial)')
    RunFit('1I',5,15,30,np.array([1e-12]),ng_acc='radial')

    print('RM 2003, no non-grav. acc.')
    RunFit('523599',10,30,70,np.array([]))
    print('RM 2003, with non-grav. acc. (RTN)')
    RunFit('523599',10,30,70,np.array([1e-12,1e-12,1e-12]),ng_acc='RTN')

    print('Kamo\'oaleva, gravity only')
    RunFit('469219',120,190,250,np.array([]))

    print('Golevka, gravity only')
    RunFit('6489',858,866,873,np.array([]),it_max=15)

