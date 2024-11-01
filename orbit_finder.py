import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import spiceypy as spice
import rebound 
import assist
from scipy.integrate import solve_ivp
from extensisq import SWAG

import config as cf


### LOAD DATA -----------------------------------------------------------------
def LoadDataMPC(obsstat_file, objdata_file, start_date=None, end_date=None):
    # load geodata of observing stations
    obss = np.array([])
    lons = np.array([])
    rcos = np.array([])
    rsin = np.array([])
    with open(obsstat_file) as f:
        f.readline() # skip header
        for l in f:
            if l[3:12] != '         ':
               obss = np.append(obss, l[0:3])
               lons = np.append(lons, float(l[3:13]))
               rcos = np.append(rcos, float(l[13:21])*cf.Re)
               rsin = np.append(rsin, float(l[21:30])*cf.Re)
    # load observational data, parsing an input file in MPC format
    OC = np.array([])
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
            # time of observation: convert to SPICE Ephemerides Time (~TDB)
            eti = spice.str2et(l[15:25]) + float(l[25:32])*cf.days
            et = np.append(et, eti)
            # observed RA
            hours, minutes, seconds = float(l[32:34]), float(l[35:37]), float(l[38:44])
            rao = hours + minutes/60 + seconds/3600
            ra = np.append(ra, np.deg2rad(15*rao)) # rad
            # observed Decl
            degrees, minutes, seconds = float(l[44:47]), float(l[48:50]), float(l[51:56])
            deo = np.sign(degrees)*(abs(degrees) + minutes/60 + seconds/3600)
            de = np.append(de, np.deg2rad(deo)) # rad
            # heliocentric position of Earth to observation point vector
            rE = spice.spkpos('399', eti, 'J2000', 'NONE', '10')[0]
            # geocentric position of observing station
            if code == '250' or code == 'C57':
                # for HST, data is in the file 
                l1 = f.readline()
                X = float(l1[34:42].replace(' ',''))
                Y = float(l1[46:54].replace(' ',''))
                Z = float(l1[58:66].replace(' ',''))
                R_ = np.array([X,Y,Z])
            elif code == '247' or code == '270':
                # "roving observer" data:
                l1 = f.readline()
                lon = np.radians(float(l1[34:44]))
                lat = np.radians(float(l1[45:55]))
                alt = float(l1[56:61])/1e3 # m -> km
                R0_ = spice.georec(lon, lat, alt, cf.Re, cf.fe)
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
            s_ra = np.append(s_ra, np.deg2rad(2./3600)*factor) # rad
            s_de = np.append(s_de, np.deg2rad(2./3600) ) # rad
    RS = np.delete(RS, 0, 0)
    # use uncertainties from Veres et al. 2017 when available:
    for key in cf.uncertainty:
        ii = np.where(OC == key)[0]
        factor = np.cos(de[ii])
        s_ra[ii] = np.deg2rad(cf.uncertainty[key]/3600)*factor # arc sec to rad 
        s_de[ii] = np.deg2rad(cf.uncertainty[key]/3600) # arc sec to rad 
    # >>> (optional) restrict epochs to the range between start_date and end_date
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
    # <<<
    # change units to AU, days 
    RS = RS / cf.AU
    et = et / cf.days
    ### return output as dictionary
    Data = {'ET':et[msk], 'OC':OC[msk], 'RS':RS[msk],
            'RA':ra[msk], 'De':de[msk], 'sigma_RA':s_ra[msk], 'sigma_De':s_de[msk]} 
    return Data 



### DIFFERENTIAL CORRECTION OF THE ORBIT --------------------------------------
# References: Farnocchia et al. (2015); Carpino et al. (2003)
def DiffCorr(Data, et0, x0, propagator, prop_args, max_iter):
    # parameters
    chi2_rec = 7.  # recovery threshold
    chi2_rej0 = 8.  # base rejection threshold
    alpha  = 0.25  # fraction of max chi-square to use as increased rejection threshold 
    frac   = 0.05  # maximum fraction of epochs discarded in a single step
    n, m = len(x0), len(Data['ET'])
    # initializations 
    x = x0
    RMS0 = 1.
    flag = np.repeat(True, m) # epochs included in fit 
    Cov = np.zeros((n, n)) 
    m_rec, m_rej, m_use = 0, 0, m
    chi2 = np.zeros(m)
    ### begin main loop of differential correction
    print('Differential correction begins.')
    print('#iter   red.chisq.    norm        ||dx||         #rec   #rej  #use frac')
    for k in range(max_iter):
        # initialize normal equations
        BTWB = np.zeros((n, n))
        BTWz = np.zeros(n)
        ress = 0.
        # get residuals and partials for current initial state vector
        zm, Bm, ym = ResidualsAndPartials(Data, et0, x, propagator, prop_args)
        for i in range(m):
            sigma = np.r_[Data['sigma_RA'][i], Data['sigma_De'][i]]
            W = np.diag(1./sigma**2)
            z = zm[i,:]
            B = Bm[i,:,:]
            # epochs flagged True are used in the fit
            if flag[i]:
                # accumulate normal equations
                BTWB += (B.T @ W @ B)
                BTWz += (B.T @ W @ z)
                ress += (z @ W @ z)/2.
            # calculate post-fit residual covariance and chi-square for epoch i
            G = np.diag(sigma**2) + (1.-2.*flag[i])*(B @ Cov @ B.T)
            chi2[i] = z @ np.linalg.inv(G) @ z
        ### update initial state vector and its covariance
        dx = np.linalg.solve(BTWB, BTWz)
        x = x + dx
        Cov = np.linalg.inv(BTWB)
        # update RMS and norm of correction
        RMS = np.sqrt(ress/m)
        nrm = np.sqrt((dx @ BTWB @ dx)/n)
        ### screen output
        print('%4i   %12.6e %12.6e %12.6e %6i %6i %6i %4.2f' % (k, RMS, nrm,
               np.linalg.norm(dx), m_rec, m_rej, m_use, m_use/m))
        # >>> this section handles outliers rejection/recovery
        # adjust rejection threshld for this step:
        chi2_rej = max( chi2_rej0 + 400.*(1.2)**(-m_use), alpha * chi2[flag].max() ) 
        # mark epochs to be readmitted in the fit at next step...
        i_rec = np.where((flag == False) & (chi2 < chi2_rec))[0]
        # ... and update their flag
        flag[i_rec] = True
        m_rec = len(i_rec)
        # mark epochs to be excluded in next step...
        i_mrk = np.where((flag == True) & (chi2 > chi2_rej))[0]
        # but only exclude up to (frac * m_use), in decreasing order of residual:
        m_rej = min(len(i_mrk), int(frac*m_use))
        chi2_mrk = np.flip(np.sort(chi2[i_mrk]))
        if len(i_mrk) > 0:
            k1 = [np.where(chi2 == chi2_m)[0][0] for chi2_m in chi2[i_mrk][:m_rej]]
            flag[k1] = False
        # update total number of epochs that will be used in the fit at next step:
        m_use = sum(flag)
        # <<<  
        # condition to end loop:
        stop = (abs(RMS-RMS0)/RMS0 < 1e-3) | (RMS < 0.5) \
               | (nrm < 0.5) | (np.linalg.norm(dx) < 1e-7)
        RMS0 = RMS
        if stop:
            break
    print('End of differential correction.')
    res = np.r_[zm[:,0], zm[:,1]]
    # output as a dictionary
    Fit = {'x':x, 'Cov':Cov, 'res':res, 'flag':flag, 'RMS':RMS, 'norm':nrm, 
           'ET0':et0, 'y':ym}
    return Fit

def ResidualsAndPartials(Data, et0, x, propagator, prop_args):
    n, m = len(x), len(Data['ET'])
    # call propagator function (its name is passed as input: "propagator"); returns
    # state vector yy, state transition matrix PP, sensitivity matrix SS at all epochs
    # Note: propagator also takes care of light travel time iteration
    yy, PP, SS = propagator(x, et0, Data['ET'], Data['RS'], prop_args)
    # initialize residuals and their partials matrix for current x
    B = np.zeros((m, 2, n))
    z = np.zeros((m, 2))
    # fill in B and z (Escobal 1976 formulae)
    for i in range(m):
        R_ = Data['RS'][i,:]
        r_, P, S = yy[i,:3], PP[i,:,:], SS[i,:,:]
        rho_ = r_ - R_
        rho = np.linalg.norm(rho_)
        cos_ra, sin_ra = np.cos(Data['RA'][i]), np.sin(Data['RA'][i])
        cos_de, sin_de = np.cos(Data['De'][i]), np.sin(Data['De'][i])
        A_ = np.array([-sin_ra, cos_ra, 0.])
        D_ = np.array([-sin_de*cos_ra, -sin_de*sin_ra, cos_de ])
        L_ = np.array([ cos_de*cos_ra,  cos_de*sin_ra, sin_de ])
        dL_ = (L_ - rho_/rho)
        # fill partials matrix
        B[i,:,:] = np.c_[ np.r_[ A_, np.zeros(3) ]/rho @ np.block([P, S]),
                          np.r_[ D_, np.zeros(3) ]/rho @ np.block([P, S]) ].T
        # fill residual vector
        z[i,:] = np.dot(dL_, A_), np.dot(dL_, D_) 
    return z, B, yy



### TRAJECTORY PROPAGATION ----------------------------------------------------
# Two propagators are available, based on REBOUND/ASSIST and SciPy solve_ivp
# FIRST OPTION: use ASSIST (faster, probably more accurate, but less flexible)
def PropagateAssist(x, et0, et, RS, forces):
    ### iteration needed to account for light travel time:
    m = len(et)
    tau = np.zeros(m)
    for j in range(2):
        yy, PP, SS = RunAssist(x, et0, et, tau, forces)
        for i in range(m):
            r_ = yy[i,:3]
            R_ = RS[i,:]
            # note speed of light in vacuum, cc, is in AU/days 
            tau[i] = np.linalg.norm(r_-R_)/cf.cc
    return yy, PP, SS
        
def RunAssist(x, et0, et, tau, forces):
    t0 = et0
    t = et - tau
    p_  = x[6:]
    # set up rebound simulation and ephemerides extension
    ephem = assist.Ephem(cf.assist_planets_file, cf.assist_asteroids_file)
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
def PropagateSciPy(x, et0, et, RS, aNG):
    rtol = 1e-11
    atol = 1e-13
    method = SWAG
    n = len(x)
    m = len(et)
    n_p = n-6  # number of parameters beyond initial state vector
    r0_, v0_, parms_ = x[0:3], x[3:6], x[6:]
    # form initial conditions for ODE integrator
    y0 = np.concatenate([r0_, v0_, np.eye(6).flatten(), np.zeros((6,n_p)).flatten()])
    # forward integration from et0 to et[-1]
    tspan_f = [et0, et[-1]]
    sol_f = solve_ivp(Derivs, tspan_f, y0, method=method, args=(parms_, aNG),
            rtol=rtol, atol=atol, dense_output=True)
    # backward integration from et0 to et[0]
    tspan_b = [et0, et[0]]
    sol_b = solve_ivp(Derivs, tspan_b, y0, method=method, args=(parms_, aNG),
            rtol=rtol, atol=atol, dense_output=True)
    ii_f = np.where(et >  et0)[0]
    ii_b = np.where(et <= et0)[0]
    ### iteration needed to account for light travel time:
    tau = np.zeros(m)
    for j in range(2):
        teval_b = et[ii_b]-tau[ii_b]
        teval_f = et[ii_f]-tau[ii_f]
        #sol = np.r_[sol_b.sol(teval_b).T, sol_f.sol(teval_f).T]
        sol = np.zeros((m, len(y0)))
        for l, tb in enumerate(teval_b):
            sol[l,:] = sol_b.sol(tb).T
        for l, tf in enumerate(teval_f):
            sol[l,:] = sol_f.sol(tf).T        

        for i in range(m):
            r_ = sol[i,:3]
            R_ = RS[i,:]
            # note speed of light in vacuum, cc, is in AU/days 
            tau[i] = np.linalg.norm(r_-R_)/cf.cc
    # prepare output
    yy = np.reshape(sol[:,  :6], (m, 6))
    PP = np.reshape(sol[:,6:42], (m, 6, 6))
    SS = np.reshape(sol[:,42: ], (m, 6, n_p))
    return yy, PP, SS

# force model and linearization: see Montenbruck & Gill 2005, Chapters 3 and 7, resp.
def Derivs(t, y, parms_,aNG):
    r_ = y[0:3]
    v_ = y[3:6]
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    # main term
    f_ = -cf.mu_s*r_/r**3
    # perturbations
    p_ = 0
    # planets
    for i in range(len(cf.mu_p)):
        s_, _ = spice.spkpos(cf.tgP[i], t*cf.days, 'J2000', 'NONE', '10')
        s_ = s_/cf.AU
        p_ = p_ - cf.mu_p[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # asteroids
    for i in range(len(cf.mu_a)):
        s_, _ = spice.spkpos(cf.tgA[i], t*cf.days, 'J2000', 'NONE', '10')
        s_ = s_/cf.AU
        p_ = p_ - cf.mu_a[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # GR correction
    p_ = p_ + (cf.mu_s/cf.cc**2/r**3)*( (4*cf.mu_s/r - v**2)*r_ + 4*np.dot(r_,v_)*v_ )
    # non-gravitational term
    if not aNG:
       aNG_ = np.zeros(3)
       dadp = np.array([])
       n_p  = 0
    else:
       aNG_, dadp = aNG(r_, v_, parms_)
       n_p = len(parms_)
    p_ = p_ + aNG_
    # total acceleration
    a_ = f_ + p_
    ### variational equations
    PHI = np.reshape(y[6:42], (6, 6))
    # main term
    F = -cf.mu_s/r**3*( np.eye(3) - 3*np.outer(r_, r_)/r**2)
    # point mass perturbers (planets only!)
    P = np.zeros((3,3))
    for i in range(len(cf.mu_p)):
        s_, _ = spice.spkpos(cf.tgP[i], t*cf.days, 'J2000', 'NONE', '10')
        s_ = s_/cf.AU
        P = P - cf.mu_p[i]/np.linalg.norm(r_-s_)**3 \
          * ( np.eye(3) - 3*np.outer(r_-s_,r_-s_)/np.linalg.norm(r_-s_)**2 )
    # asteroids, relativity, and radiation pressure are omitted in the variational eqns.
    G = F + P
    AP = np.block([[np.zeros((3, 3)), np.eye(3)], [G, np.zeros((3, 3))]]) @ PHI
    # sensitivity matrix
    if n_p > 0:
        S = np.reshape(y[42:], (6, n_p))
        AS = np.block([[np.zeros((3, 3)), np.eye(3)], [G, np.zeros((3, 3))]]) @ S \
           + np.concatenate((np.zeros((3, n_p)),np.c_[dadp]))
    else:
        AS = np.array([])
    ### full vector with derivatives 
    dydt = np.concatenate([v_, a_, AP.flatten(), AS.flatten()])
    return dydt



### OUTPUT --------------------------------------------------------------------
def SummaryPlot(object_name, Data, Fit, scaled=True):
    et, s_ra, s_de, RS = Data['ET'], Data['sigma_RA'], Data['sigma_De'], Data['RS'] 
    x, z, flag, y = Fit['x'], Fit['res'], Fit['flag'], Fit['y'] 
    m = len(et)
    if scaled:
       res_ra, res_de, unit, ylim = z[:m] / s_ra, z[m:] / s_de, '($\\sigma$)', (-5, 5)     
    else:
       res_ra, res_de, unit, ylim = z[:m] * 206265., z[m:] * 206265., '(\")', (-10, 10)
    not_flag = np.logical_not(flag)
    tt = np.array([datetime.fromisoformat(spice.et2utc(eti*cf.days,'ISOC', 0)) for eti in et])
    dh = np.linalg.norm(y[:,:3], axis=1)
    rE, _ = spice.spkpos('399', et*cf.days, 'J2000', 'NONE', '10')
    dg = np.linalg.norm(y[:,:3]-rE/cf.AU, axis=1)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ### make plot
    fig, axes = plt.subplot_mosaic('AAA.;BBBY;CCCZ', figsize=(7, 7), constrained_layout=True)
    # plot heliocentric and geocentric distance
    axes['A'].plot(tt, dh, '.', label='heliocentric')
    axes['A'].plot(tt, dg, 's', ms=3, label='geocentric')
    axes['A'].xaxis.set_ticklabels([])
    axes['A'].set_ylabel('Distance (AU)')
    axes['A'].set_ybound(lower=0.)
    axes['A'].legend()  
    # plot residuals in RA 
    axes['B'].plot(tt[flag], res_ra[flag], '.', color='#00356B', label='included')
    axes['B'].plot(tt[not_flag], res_ra[not_flag], 'x', ms=4, color='darkgray', label='excluded')
    axes['B'].axhline(color='k', lw=0.7)
    axes['B'].xaxis.set_ticklabels([])
    axes['B'].set_ylabel('R. A. Res. '+unit)
    axes['B'].set_ylim(ylim)
    axes['B'].legend()
    axes['B'].grid()
    # plot residuals in DE
    axes['C'].plot(tt[flag], res_de[flag], '.', color='#00356B', label='Decl.')
    axes['C'].plot(tt[not_flag], res_de[not_flag], 'x', ms=4, color='darkgray')
    axes['C'].axhline(color='k', lw=0.7)
    axes['C'].set_ylabel('Decl. Res. '+unit)
    axes['C'].set_xlabel('Date (UTC)')
    axes['C'].xaxis.set_major_locator(locator)
    axes['C'].xaxis.set_major_formatter(formatter)
    axes['C'].set_ylim(ylim)
    axes['C'].grid()
    # add histograms
    #axes['X'].set_axis_off()
    axes['Y'].hist(res_ra[flag], bins=20, orientation='horizontal', color='#00356B')
    axes['Y'].axhline(color='k', lw=0.7)
    axes['Y'].set_ylim(ylim)
    axes['Y'].grid()
    axes['Z'].hist(res_de[flag], bins=20, orientation='horizontal', color='#00356B')
    axes['Z'].axhline(color='k', lw=0.7)
    axes['Z'].set_xlabel('Count')
    axes['Z'].set_ylim(ylim)
    axes['Z'].grid()
    plt.savefig('summary_plot.pdf')
    plt.close()

def SummaryText(object_name, Data, Fit):
    n = len(Fit['x']) 
    xs1 = ['x ', 'y ', 'z ', 'vx', 'vy', 'vz', 'A1', 'A2', 'A3']
    xs2 = np.r_[np.repeat('AU', 3), np.repeat('AU/d', 3), np.repeat('AU/d^2', 3)]
    ee1 = ['a', 'e', 'I', 'Ω', 'ω', 'M']
    ee2 = ['AU', ' ', 'deg', 'deg', 'deg', 'deg']
    U = spice.pxform( 'J2000', 'ECLIPJ2000', Fit['ET0']*cf.days )
    r_, v_ = U @ Fit['x'][0:3],   U @ Fit['x'][3:6]
    OE = spice.oscltx(np.r_[r_, v_], Fit['ET0']*cf.days, cf.mu_s)
    oe = np.r_[ OE[9], OE[1], np.rad2deg(OE[2:6]) ] 
    A = JacobianTBP(r_, v_, cf.mu_s)[1]
    Cov_oe = A @ Fit['Cov'][:6,:6] @ A.T
    sigma_oe = np.sqrt(np.diagonal(Cov_oe))
    sigma_oe[2:] *= 180/np.pi 
    with open('summary_fit.txt', 'w') as f:
        f.write('Orbit_Finder - '+datetime.today().strftime('%Y-%m-%d %H:%M:%S')+'\n\n')
        if 'C_' in object_name:
            f.write(object_name.replace('C_','C/').replace('_',' ')+'\n')
        else:
            f.write('Object name: '+object_name+'\n')
        f.write('\n')
        f.write('RMS = %8.3f\n' % Fit['RMS'])
        f.write('\n')
        f.write('# epochs = %4i \n' % sum(Fit['flag']))
        f.write('First epoch: '+spice.et2utc(Data['ET'][ 0]*cf.days, 'C', 2)+'\n')
        f.write('Last  epoch: '+spice.et2utc(Data['ET'][-1]*cf.days, 'C', 2)+'\n')
        f.write('\n')        
        f.write('Fit Epoch  : '+spice.et2utc(Fit['ET0']*cf.days, 'C', 2)+'\n')
        f.write('\n')
        f.write('State vector (J2000 frame)\n')
        for i, x_i in enumerate(Fit['x']):
            if i < 6:
                scale = 1.
            else:
                scale = 1e8
            f.write('%s = %13.9f ± %13.10f %s\n' % (xs1[i], x_i*scale, 
                    np.sqrt(Fit['Cov'][i,i])*scale, xs2[i]))            
        f.write('\n')
        f.write('Orbital elements (ECLIPJ2000 frame)\n')          
        for i in range(6):
            f.write('%s = %13.9f ± %13.10f %s\n' % (ee1[i], oe[i], sigma_oe[i], ee2[i]))
        f.write('\n')

### Get Jacobian of classical orbital elements with respect to state vector, and its inverse
### Ref.: Broucke (1970), Montenbruck & Gill (2012); modified to work for e < 1 as well as e > 1 
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
    #P_, Q_, W_ = spice.rotate(w, 3) @ spice.rotate(I, 1) @ spice.rotate(W, 3)
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



### PRELIMINARY ORBIT DETERMINATION -------------------------------------------
def InitOrbDet(et, ra, de, s_ra, s_de, RS, i1, i2, i3):
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
def AnglesOnlyIOD(taui, Ri_, Li_, mu_s, r2i, kmax, tol):
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
