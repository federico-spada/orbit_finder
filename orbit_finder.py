import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import spiceypy as spice
import rebound 
import assist
from scipy.integrate import solve_ivp
from extensisq import SWAG
import seaborn as sns

# constants 
Re = 6378.1366 # km
fe = 1./298.257223563 ### Earth oblateness, WGS84 value 
AU   = 1.495978707e8 # km
cc = 299792.458 # km/s  
days = 86400 # s 
mu_s = 132712440041.279419 # km^3/s^2

# for propagation with SciPy ode solver:
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

# for propagation with rebound/assist:
all_forces = ['SUN', 'PLANETS', 'ASTEROIDS', 'NON_GRAVITATIONAL',
              'EARTH_HARMONICS', 'SUN_HARMONICS', 'GR_EIH']
assist_path = '/Users/fs255/rebound_assist/data/'
    
# use everywhere units of AU, days
mu_s = mu_s * days**2/AU**3
mu_p = mu_p * days**2/AU**3
mu_a = mu_a * days**2/AU**3
cc = cc * days/AU

# uncertainties for a selection of observatories estimated by Veres et al. 2017
uncertainty = { '703': 1.0 , '691': 0.6 , '644': 0.6 ,
                '704': 1.0 , 'G96': 0.5 , 'F51': 0.2 , 'G45': 0.6 , '699': 0.8 ,
                'D29': 0.75, 'C51': 1.0 , 'E12': 0.75, '608': 0.6 , 'J75': 1.0 ,
                '645': 0.3 , '673': 0.3 , '689': 0.5 , '950': 0.5 , 'H01': 0.3 ,
                'J04': 0.4 , 'G83': 0.3 , 'K92': 0.4 , 'K93': 0.4 , 'Q63': 0.4 ,
                'Q64': 0.4 , 'V37': 0.4 , 'W84': 0.4 , 'W85': 0.4 , 'W86': 0.4 ,
                'W87': 0.4 , 'K91': 0.4 , 'E10': 0.4 , 'F65': 0.4 , 'Y28': 0.3 ,
                '568': 0.25, 'T09': 0.1 , 'T12': 0.1 , 'T14': 0.1 , '309': 0.3 ,
                '250': 0.05, 'C57': 0.05 } # add HST, TESS


### LOAD DATA -----------------------------------------------------------------
def LoadDataMPC(obsstat_file,objdata_file,start_date=None,end_date=None):
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
            if code == '250' or code == 'C57':
                # for HST, data is in the file 
                l1 = f.readline()
                X = float(l1[34:42].replace(' ',''))
                Y = float(l1[46:54].replace(' ',''))
                Z = float(l1[58:66].replace(' ',''))
                R_ = np.array([X,Y,Z])
            elif code == '247' or code == '270':
                # roving observer data:
                l1 = f.readline()
                lon = np.radians(float(l1[34:44]))
                lat = np.radians(float(l1[45:55]))
                alt = float(l1[56:61])/1e3 # m -> km
                R0_ = spice.georec(lon, lat, alt, Re, fe)
                U = spice.pxform( 'ITRF93', 'J2000', eti )
                R_ = U @ R0_
            else:
                # for other observatories, must be calculated from geodetic data
                io = np.where(obss == code)[0]
                R0_ = spice.cylrec(rcos[io], np.deg2rad(lons[io]), rsin[io])
                # transform from e.f. to s.f. coordinates
                U = spice.pxform( 'ITRF93', 'J2000', eti )
                R_ = U @ R0_
            RS = np.vstack((RS, R_+rE))
            # uncertainties in RA, Decl: assume 2 arc sec
            factor = np.cos(np.deg2rad(deo))
            s_ra = np.append( s_ra, np.deg2rad(2./3600)*factor ) # rad
            s_de = np.append( s_de, np.deg2rad(2./3600) ) # rad
    RS = np.delete(RS, 0, 0)
    # use information from dictionary of uncertainties:
    for key in uncertainty:
        ii = np.where(OC == key)[0]
        factor = np.cos(de[ii])
        s_ra[ii] = np.deg2rad(uncertainty[key]/3600)*factor # arc sec to rad 
        s_de[ii] = np.deg2rad(uncertainty[key]/3600) # arc sec to rad 
    # (optional) restrict epochs to the range between start_date and end_date
    if not start_date:
        start_date = spice.et2utc(et[0],'ISOC',0)
        i_start = 0
    else:
        et_start = spice.str2et(start_date)
        i_start = np.where(et >= et_start)[0][0]
    if not end_date:
        end_date = spice.et2utc(et[-1],'ISOC',0)
        i_end = len(et)        
    else:
        et_end = spice.str2et(end_date)
        i_end = np.where(et <= et_end)[0][-1]
    print('Using data between', start_date, ' and', end_date, 
          '; range:', i_start, '-' ,i_end)
    # all epochs outside this range will be dropped:
    msk = range(i_start,i_end)
    # change units to AU, days 
    RS = RS / AU
    et = et / days
    return et[msk], ra[msk], de[msk], s_ra[msk], s_de[msk], RS[msk], JD[msk], OC[msk]



### PRELIMINARY ORBIT DETERMINATION -------------------------------------------
def InitOrbDet(et,ra,de,s_ra,s_de,RS,i1,i2,i3):
    taui = et[i1]-et[i2], et[i3]-et[i2], et[i3]-et[i1]
    Ri_ = RS[i1,:], RS[i2,:], RS[i3,:]
    e1_ = spice.radrec(1.,ra[i1],de[i1])
    e2_ = spice.radrec(1.,ra[i2],de[i2])
    e3_ = spice.radrec(1.,ra[i3],de[i3])
    ei_ = e1_, e2_, e3_
    r2i = 3.0
    kmax = 50
    tol = 1e-6
    r2_, v2_, k = AnglesOnlyIOD(taui,Ri_,ei_,mu_s,r2i,kmax,tol)
    if k < kmax-1:
       print('Preliminary orbit determination converged in %i iterations' % k)
       exit_code = 0
    else:
       print('WARNING: Preliminary orbit determination did not converge in %i iterations' % kmax)
       exit_code = 1
    return r2_, v2_, exit_code

# Reference: Bate, Mueller, White (1971), Section 5.8, page 271
def AnglesOnlyIOD(taui,Ri_,Li_,mu_s,r2i,kmax,tol):
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
def DiffCorr(et,ra,de,s_ra,s_de,RS,et0,x0,propagator,prop_args,max_iter):
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
    z, B = ResidualsAndPartials(et,ra,de,RS,et0,x,propagator,prop_args)
    Cov = np.linalg.inv(B.T @ W @ B)
    flag = np.repeat(True,m) # True for epochs included in the fit
    X2 = np.zeros(m)
    m_use = m
    # begin differential correction iteration
    print('Differential correction begins.')
    print('#iter   red.chisq.   metric      ||dx/x||        #rec   #rej  #use frac')
    for k in range(max_iter):
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
        z, B = ResidualsAndPartials(et,ra,de,RS,et0,x,propagator,prop_args)
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
            k1 =[np.where(X2 == X2_m)[0][0] for X2_m in X2[i_mrk][:m_rej]]
            flag[k1] = False
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

def ResidualsAndPartials(et,ra,de,RS,et0,x,propagator,prop_args):
    n = len(x)
    m = len(et)
    # call propagator function; its name is passed as input ("propagator")
    # Note: "propagator" also takes care of light travel time iteration
    yy, PP, SS = propagator(x,et0,et,RS,prop_args)
    # initialize residuals and their partials matrix for current x
    B = np.zeros((2*m,n))
    z = np.zeros(2*m)
    # fill in B and z (Escobal 1976 formulae)
    for i in range(m):
        r_ = yy[i,:3]
        R_ = RS[i,:]
        P  = PP[i,:,:]
        S  = SS[i,:,:]
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



### TRAJECTORY PROPAGATION ----------------------------------------------------
# Two propagators are available, based on REBOUND/ASSIST and SciPy solve_ivp
# FIRST OPTION: use ASSIST (faster, probably more accurate, but less flexible)
def PropagateAssist(x,et0,et,RS,forces):
    ### iteration needed to account for light travel time:
    m = len(et)
    tau = np.zeros(m)
    for j in range(2):
        yy, PP, SS = RunAssist(x,et0,et,tau,forces)
        for i in range(m):
            r_ = yy[i,:3]
            R_ = RS[i,:]
            # note speed of light in vacuum, cc, is in AU/days 
            tau[i] = np.linalg.norm(r_-R_)/cc
    return yy, PP, SS
        
def RunAssist(x,et0,et,tau,forces):
    t0 = et0
    t = et - tau
    p_  = x[6:]
    # set up rebound simulation and ephemerides extension
    ephem = assist.Ephem(assist_path+'linux_p1550p2650.440',
                         assist_path+'sb441-n16.bsp')
    sim = rebound.Simulation()
    # initial position of the Sun
    sun0 = ephem.get_particle("sun", t0)
    # initial conditions of test particle
    part0_h = rebound.Particle( x = x[0],  y = x[1],  z = x[2],
                               vx = x[3], vy = x[4], vz = x[5] )
    # change from heliocentric to SS barycentric frame
    part0 = sun0 + part0_h
    # initialize simulation 
    sim.add(part0)
    sim.t = t0
    extras = assist.Extras(sim, ephem)
    # parameters for non-gravitational forces
    nparms = len(p_)
    params_ngforce = np.zeros(3)
    for k in range(nparms):
        params_ngforce[k] = p_[k]
    extras.particle_params = params_ngforce
    # >>> list with forces to be included
    extras.forces = forces
    #print(extras.forces) # check what is included
    # <<<
    # prepare for integration 
    m = len(t)
    y = np.zeros((m,6))
    ### first integration: get state vector with nominal values of p_
    for i, ti in enumerate(t):
        extras.integrate_or_interpolate(ti)
        ref = ephem.get_particle("sun", ti)
        y[i,0:3] = np.array(sim.particles[0].xyz) - np.array(ref.xyz)
        y[i,3:6] = np.array(sim.particles[0].vxyz)
    sim = None
    ### set up integration with varying p_ components, to evaluate the
    ### sensitivity matrix
    eps = 1e-6
    S = np.zeros((m,6,nparms))
    y1 = np.zeros(6)
    for k in range(nparms):
        delta_params_ngforce = np.zeros(3)
        delta_params_ngforce[k] = params_ngforce[k] * eps
        # initialize simulation 
        sim = rebound.Simulation()
        sim.add(part0)
        sim.t = t0
        extras = assist.Extras(sim, ephem)
        extras.particle_params = params_ngforce + delta_params_ngforce
        for i, ti in enumerate(t):
            extras.integrate_or_interpolate(ti)
            ref = ephem.get_particle("sun", ti)
            y1[0:3] = np.array(sim.particles[0].xyz) - np.array(ref.xyz)
            y1[3:6] = np.array(sim.particles[0].vxyz)
            for j in range(6):
                S[i,j,k] = (y1[j] - y[i,j])/(delta_params_ngforce[k])
        sim = None
    ##
    ### To evaluate state transition matrix use variational particles
    ### Note: for some reason, this integration must be done separately
    ### I am not really sure why, but it must be due to the way variational
    ### particles are treated in REBOUND/ASSIST
    # initialize simulation 
    sim = rebound.Simulation()
    sim.add(part0)
    sim.t = t0
    extras = assist.Extras(sim, ephem)
    # turn off non-gravitational forces to use variational particles
    #forces = extras.forces
    #forces.remove("NON_GRAVITATIONAL")
    #extras.forces = forces
    # add variational particles to calculate state transition matrix
    vp_x0 = sim.add_variation(testparticle=0,order=1)
    vp_x0.particles[0].x = 1
    vp_y0 = sim.add_variation(testparticle=0,order=1)
    vp_y0.particles[0].y = 1
    vp_z0 = sim.add_variation(testparticle=0,order=1)
    vp_z0.particles[0].z = 1
    vp_vx0 = sim.add_variation(testparticle=0,order=1)
    vp_vx0.particles[0].vx = 1
    vp_vy0 = sim.add_variation(testparticle=0,order=1)
    vp_vy0.particles[0].vy = 1
    vp_vz0 = sim.add_variation(testparticle=0,order=1)
    vp_vz0.particles[0].vz = 1
    # prepare for integration 
    m = len(t)
    P = np.zeros((m,6,6))
    # run rebound simulation + assist
    for i, ti in enumerate(t):
        extras.integrate_or_interpolate(ti)
        # state transition matrix
        P[i,:,0] = np.r_[ vp_x0.particles[0].xyz,  vp_x0.particles[0].vxyz ]
        P[i,:,1] = np.r_[ vp_y0.particles[0].xyz,  vp_y0.particles[0].vxyz ]
        P[i,:,2] = np.r_[ vp_z0.particles[0].xyz,  vp_z0.particles[0].vxyz ]
        P[i,:,3] = np.r_[ vp_vx0.particles[0].xyz, vp_vx0.particles[0].vxyz ]
        P[i,:,4] = np.r_[ vp_vy0.particles[0].xyz, vp_vy0.particles[0].vxyz ]
        P[i,:,5] = np.r_[ vp_vz0.particles[0].xyz, vp_vz0.particles[0].vxyz ]
    return y, P, S

# SECOND OPTION: integrate the equations of motion with the SciPy ode solver
# "solve_ivp"; slower option, but fully customizable equations (see "Derivs")
def PropagateSciPy(x,et0,et,RS,aNG):
    rtol = 1e-11
    atol = 1e-13
    n = len(x)
    m = len(et)
    n_p = n-6  # number of parameters beyond initial state vector
    r0_, v0_, parms_ = x[0:3], x[3:6], x[6:]
    # form initial conditions for ODE integrator
    y0 = np.concatenate([r0_, v0_, np.eye(6).flatten(), np.zeros((6,n_p)).flatten()])
    # forward integration from et0 to et[-1]
    tspan_f = [et0, et[-1]]
    sol_f = solve_ivp(Derivs,tspan_f,y0,method=SWAG,args=(parms_,aNG),
            rtol=rtol,atol=atol,dense_output=True)
    # backward integration from et0 to et[0]
    tspan_b = [et0, et[0]]
    sol_b = solve_ivp(Derivs,tspan_b,y0,method=SWAG,args=(parms_,aNG),
            rtol=rtol,atol=atol,dense_output=True)
    ii_f = np.where(et >  et0)[0]
    ii_b = np.where(et <= et0)[0]
    ### iteration needed to account for light travel time:
    tau = np.zeros(m)
    for j in range(2):
        teval_b = et[ii_b]-tau[ii_b]
        teval_f = et[ii_f]-tau[ii_f]
        sol = np.r_[sol_b.sol(teval_b).T, sol_f.sol(teval_f).T]
        for i in range(m):
            r_ = sol[i,:3]
            R_ = RS[i,:]
            # note speed of light in vacuum, cc, is in AU/days 
            tau[i] = np.linalg.norm(r_-R_)/cc
    # prepare output
    yy = np.reshape(sol[:,  :6],(m,6))
    PP = np.reshape(sol[:,6:42],(m,6,6))
    SS = np.reshape(sol[:,42: ],(m,6,n_p))
    return yy, PP, SS

# force model and linearization: see Montenbruck & Gill 2005, Chapters 3 and 7, resp.
def Derivs(t,y,parms_,aNG):
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
    # include Earth oblateness term - experimental, use with caution
    #J2 = 1.08262539e-3
    #J2 = 0.00108263
    #C = 1.5*J2*mu_p[2]*(6378.1366/AU)**2
    #s_, _ = spice.spkpos('399', t*days, 'J2000', 'NONE', '10')
    #U = spice.pxform('J2000', 'ITRF93', t*days)
    #r1_ = U @ (r_ - s_/AU) # geocentric radius vector, and convert to ECEF frame
    #r1 = np.linalg.norm(r1_)
    #x1, y1, z1 = r1_
    #a1_ = (C/r1**4) * np.array([ (5*(z1/r1)**2 - 1)*x1/r1,
    #                             (5*(z1/r1)**2 - 1)*y1/r1,
    #                             (5*(z1/r1)**2 - 3)*z1/r1 ])
    #aJ2_ = U.T @ a1_ # convert acceleration back to J2000 frame before adding it up
    #p_ = p_ + aJ2_
    # non-gravitational term
    if not aNG:
       aNG_ = np.zeros(3)
       dadp = np.array([])
       n_p  = 0
    else:
       aNG_, dadp = aNG(r_,v_,parms_)
       n_p = len(parms_)
    p_ = p_ + aNG_
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
    if n_p > 0:
        S = np.reshape(y[42:],(6,n_p))
        AS = np.block([[np.zeros((3,3)), np.eye(3)], [G, np.zeros((3,3))]]) @ S \
           + np.concatenate((np.zeros((3,n_p)),np.c_[dadp]))
    else:
        AS = np.array([])
    ### full vector with derivatives 
    dydt = np.concatenate([v_, a_, AP.flatten(), AS.flatten()])
    return dydt


# PLOTTING OUTPUT -------------------------------------------------------------
def PlotResiduals(et,z,s_ra,s_de,RS,et0,propagator,prop_args,flag,x,title):
    res_ra = z[:len(et)]
    res_de = z[len(et):]
    not_flag = np.logical_not(flag)
    tt = np.array([datetime.fromisoformat(spice.et2utc(eti*days,'ISOC', 0)) 
                  for eti in et])
    yy, _, _ = propagator(x,et0,et,RS,prop_args)
    dd = np.linalg.norm(yy[:,:3],axis=1)
    tmin = datetime.fromisoformat(spice.et2utc(min(et)*days,'ISOC', 0))
    tmax = datetime.fromisoformat(spice.et2utc(max(et)*days,'ISOC', 0))
    # plot heliocentric distance
    plt.subplot(311)
    plt.title(title)
    plt.plot(tt[flag],dd[flag],'.',color='#00356B')
    plt.plot(tt[not_flag],dd[not_flag],'x',ms=4,color='darkgray')
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().set_xbound(tmin, tmax)
    plt.ylabel('Helioc. Dist. (AU)')
    plt.grid()
    # plot residuals in RA 
    plt.subplot(312)
    plt.plot(tt[flag],3600*np.rad2deg(res_ra[flag]),'.',color='#00356B',label='R.A.')
    plt.plot(tt[not_flag],3600*np.rad2deg(res_ra[not_flag]),'x',ms=4,color='darkgray')
    plt.ylabel('R. A. Res. (\")')
    plt.ylim(min(3600*np.rad2deg(res_ra[flag])),max(3600*np.rad2deg(res_ra[flag])))
    plt.plot([tmin,tmax],[0,0],'r-',lw=0.7)
    plt.grid()
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().set_xbound(tmin, tmax)
    # plot residuals in DE
    plt.subplot(313)
    plt.plot(tt[flag],3600*np.rad2deg(res_de[flag]),'.',color='#00356B',label='Decl.')
    plt.plot(tt[not_flag],3600*np.rad2deg(res_de[not_flag]),'x',ms=4,color='darkgray')
    plt.ylabel('Decl. Res. (\")')
    plt.ylim(min(3600*np.rad2deg(res_de[flag])),max(3600*np.rad2deg(res_de[flag])))
    plt.plot([tmin,tmax],[0,0],'r-',lw=0.7)
    plt.xlabel('Date (UTC)')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().set_xbound(tmin, tmax)
    plt.grid()
    plt.tight_layout()

def PlotHistRes(z,s_ra,s_de,flag,title):     
    res_ra = z[:len(flag)]
    res_de = z[len(flag):]
    plt.subplot(121)
    plt.title(title)
    sns.histplot(res_ra[flag]/s_ra[flag],kde=True,bins=20)
    plt.xlabel('R. A. Res. (\")')
    plt.ylabel('Count')
    plt.subplot(122)
    plt.title(title)
    sns.histplot(res_de[flag]/s_de[flag],kde=True,bins=20,color='C1')
    plt.xlabel('Decl. Res. (\")')
    plt.ylabel('Count')
    plt.tight_layout()

