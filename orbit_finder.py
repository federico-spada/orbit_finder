import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import spiceypy as spice
import rebound 
import assist
from scipy.integrate import solve_ivp
from extensisq import SWAG
from astropy import units as u
from astropy_healpix import HEALPix

import config as cf


### LOAD DATA -----------------------------------------------------------------
def LoadDataMPC(obsstat_file, objdata_file, start_date=None, end_date=None):
    ### load geodata of observing stations
    with open(obsstat_file, 'r') as file:
        file.readline() # skip header
        lines = [line.strip() for line in file.readlines() \
                 if (line.strip()[3:12] != '         ')]
    obss = np.array([line[0:3] for line in lines])
    lons = np.array([float(line[3:13]) for line in lines])
    rcos = np.array([float(line[13:21])*cf.RE for line in lines])
    rsin = np.array([float(line[21:30])*cf.RE for line in lines])
    ### parse data from input file in MPC format
    # read MPC file
    with open(objdata_file, 'r') as file:
        lines = file.readlines()
    # parse lines
    Nmax = len(lines) 
    et, ra, de = np.empty((3, Nmax))
    RS = np.empty((Nmax, 3))
    prg_code, obs_type, cat_code = np.empty((3, Nmax), dtype='U1')
    obs_code = np.empty(Nmax, dtype='U3')
    count = 0
    j = 0
    while count < len(lines):
        line = lines[count]        
        # program code
        prg_code[j] = line[13]
        # observation type
        obs_type[j] = line[14]
        # observing epoch parsed into SPICE's Ephemeris Time (~TDB)
        et[j] = spice.str2et(line[15:25]) + float(line[25:32]) * cf.DAYS
        # right ascension (converted to radians)
        ra[j] = np.deg2rad(15. * float(line[32:34]) + float(line[35:37]) / 4.
                                                    + float(line[38:44]) / 240.)
        # declination (converted to radians)
        de[j] = np.deg2rad(float(line[44]+'1') * (abs(float(line[45:47]))
                                                    + float(line[48:50]) / 60.
                                                    + float(line[51:56]) / 3600.))
        # astrometric catalog code:
        cat_code[j] = line[71]
        # observing station code
        obs_code[j] = line[77:80]
        # heliocentric position of obs. station (converted from ECEF to ECI, and from geoc. to helioc.)
        rE = spice.spkpos('399', et[j], 'J2000', 'NONE', '10')[0]
        if obs_type[j] in ['s', 'S']:
            # space telescopes observations: second line contains J2000 position of telescope
            next_line = lines[count+1]
            xyz = next_line[34:].split()
            RS[j] = np.r_[float(xyz[0]+xyz[1]), float(xyz[2]+xyz[3]), float(xyz[4]+xyz[5])] + rE
            increment = 2
        elif obs_type[j] in ['v', 'V']:
            # roving observer: second lind contains geodata of observing location
            next_line = lines[count+1]
            lon, lat, alt = [float(x) for x in next_line[34:].split()[:3]]
            R0 = spice.georec(np.radians(lon), np.radians(lat), alt/1e3, cf.RE, cf.FE)
            U = spice.pxform( 'ITRF93', 'J2000', et[j] )
            RS[j] = U @ R0 + rE
            increment = 2
        else:
            # "standard case": look up geodata in MPC list
            indx_obs = np.where(obss == obs_code[j])[0]
            R0 = spice.cylrec(rcos[indx_obs], np.deg2rad(lons[indx_obs]), rsin[indx_obs])
            U = spice.pxform( 'ITRF93', 'J2000', et[j] )
            RS[j] = U @ R0 + rE
            increment = 1
        count += increment
        j += 1
    # trim to actual size
    et = et[:j]; ra = ra[:j]; de = de[:j]; RS = RS[:j,:]; prg_code = prg_code[:j]
    obs_type = obs_type[:j]; cat_code = cat_code[:j]; obs_code = obs_code[:j]
    # add sigmas all equal to a default value (e.g., 2 arc sec)
    sigma_ra = np.repeat(np.deg2rad(2/3600), j) * np.cos(de)
    sigma_de = np.repeat(np.deg2rad(2/3600), j)
    print('Parsed observations at %5i epochs' % j)
    # optional: use only observations within given time frame [start_date, end_date]
    if not start_date:
        start_date = spice.et2utc(et[0], 'ISOC', 7)
        i_start = 0
    else:
        et_start = spice.str2et(start_date)
        i_start = np.where(et >= et_start)[0][0]
    if not end_date:
        end_date = spice.et2utc(et[-1], 'ISOC', 7)
        i_end = len(et)
    else:
        et_end = spice.str2et(end_date)
        i_end = np.where(et <= et_end)[0][-1]
    print('Using data between', start_date, ' and', end_date, '; range:', i_start, '-' ,i_end)
    # observations at epochs outside this range will be dropped:
    mask = range(i_start, i_end)
    # <<<
    # save epochs as datetime objects for later use
    obs_epoch = np.array([datetime.fromisoformat(spice.et2utc(eti, 'ISOC', 7)) for eti in et])
    # change units to AU, days 
    RS = RS / cf.AU
    et = et / cf.DAYS
    ### return output as dictionary
    Data = {'ET':et[mask], 'RA':ra[mask], 'De':de[mask], 'RS':RS[mask],
            'sigma_RA':sigma_ra[mask], 'sigma_De':sigma_de[mask],
            'Obs.Epoch':obs_epoch[mask], 'Prg.Code':prg_code[mask], 'Obs.Type':obs_type[mask],
            'Catalog':cat_code[mask], 'Obs.Code':obs_code[mask]}
    return Data

# Debiasing: Eggl et al. (2020)
def DebiasData(bias_file, Data):
    with open(bias_file, 'r') as file:
        lines = file.readlines()[:5]
    nside = int(lines[1][9:11])
    hp = HEALPix(nside=nside)
    catalogs = lines[4][1:].strip().split()
    bias = np.loadtxt(bias_file, skiprows=23)
    cid = [catalogs.index(cc) if cc in catalogs else -1 for cc in Data['Catalog']]
    pid = [hp.lonlat_to_healpix(ra * u.rad, de * u.rad) for ra, de in zip( Data['RA'], Data['De'])]
    dRA, dDE, pmRA, pmDE = np.array( [bias[pid[i], 4*cid[i]:4*cid[i]+4] if cid[i] != -1
                                  else [0, 0, 0, 0] for i in range(len(Data['ET']))] ).T
    count = sum([c >= 0 for c in cid])
    print('Bias correction applied to astrometric data at %i epochs.' % count)
    dt = Data['ET']/365.25
    delta_ra = (dRA + dt * pmRA/1e3) / np.cos(Data['De'])
    delta_de =  dDE + dt * pmDE/1e3
    Data['RA'] = Data['RA'] - np.deg2rad(delta_ra/3600.)
    Data['De'] = Data['De'] - np.deg2rad(delta_de/3600.)
    return Data

# Weighting scheme: Vereš et al. (2017)
def AssignUncertainties(Data, default_sigma = 2.0):
    # The following observation types are not explicitly addressed in  Vereš et al. (2017),
    #    and will be assigned the global default value: 
    other_types = ['V/v', 'R/r', 'c', 'D', 'Z', 'W/w', 'Q/q', 'T/t']
    i_other_types = np.isin(Data['Obs.Type'], other_types)
    if np.any(Data['Obs.Type'][i_other_types]):
        print('Warning: found some observation types which are not included in weighting scheme:')
        print(np.unique(Data['Obs.Type'][i_other_types]))
        print('Default sigma = %5.2f arc sec will be used.' % default_sigma)
    # initialization with _global_ default value:
    sigma = np.repeat(default_sigma, len(Data['ET']))
    ### CCD observations
    # "other" CCD obs., w/ and w/o catalog info, cf. Table 3:
    i_CCD = (Data['Obs.Type'] == 'C')
    sigma[i_CCD] = np.where(Data['Catalog'][i_CCD] != ' ', 1.0, 1.5)
    # process CMOS (obs. type "B") the same way as generic CCD
    i_CMOS = (Data['Obs.Type'] == 'B')
    sigma[i_CMOS] = np.where(Data['Catalog'][i_CMOS] != ' ', 1.0, 1.5)
    # Table 2
    i_703 = (Data['Obs.Code'] == '703') # Catalina
    sigma[i_703] = np.where(Data['ET'][i_703]*cf.DAYS < spice.str2et('2014-01-01'), 1.0, 0.8)
    i_691 = (Data['Obs.Code'] == '691') # Spacewatch
    sigma[i_691] = np.where(Data['ET'][i_691]*cf.DAYS < spice.str2et('2003-01-01'), 0.6, 0.5)
    i_644 = (Data['Obs.Code'] == '644') # NEAT
    sigma[i_644] = np.where(Data['ET'][i_644]*cf.DAYS < spice.str2et('2003-09-01'), 0.6, 0.4)
    # Table 3 (except for "other", see above), and Table 4 (except for G83, Y28, 568, T09, T12, 
    # T14, 309; see below)
    Tables_3_4 = {'704': 1.0 , 'G96': 0.5 , 'F51': 0.2 , 'G45': 0.6 , '699': 0.8 ,
                  'D29': 0.75, 'C51': 1.0 , 'E12': 0.75, '608': 0.6 , 'J75': 1.0 ,
                  '645': 0.3 , '673': 0.3 , '689': 0.5 , '950': 0.5 , 'H01': 0.3 ,
                  'J04': 0.4 , 'K92': 0.4 , 'K93': 0.4 , 'Q63': 0.4 ,
                  'Q64': 0.4 , 'V37': 0.4 , 'W84': 0.5 , 'W85': 0.4 , 'W86': 0.4 ,
                  'W87': 0.4 , 'K91': 0.4 , 'E10': 0.4 , 'F65': 0.4}
    for ocode, value in Tables_3_4.items():
        sigma[Data['Obs.Code'] == ocode] = value
    # Table 4, special cases
    gaia = ['U', 'V', 'W', 'X', '3', '6']
    ## convenience function 
    def apply_sigma_based_on_catalog(data, sigma, obs_codes, catalogs, prg_code=None, value=None):
        condition = np.isin(data['Obs.Code'], obs_codes) & np.isin(data['Catalog'], catalogs)
        if prg_code:
            condition &= (data['Prg.Code'] == prg_code)
        sigma[condition] = value
    ##
    # G83
    apply_sigma_based_on_catalog(Data, sigma, 'G83', ['q', 't'], prg_code='2', value=0.3)
    apply_sigma_based_on_catalog(Data, sigma, 'G83', gaia      , prg_code='2', value=0.2)
    # Y28
    apply_sigma_based_on_catalog(Data, sigma, 'Y28', gaia + ['t'], value=0.3)
    # 568
    apply_sigma_based_on_catalog(Data, sigma, '568', ['o', 's'], value=0.5)
    apply_sigma_based_on_catalog(Data, sigma, '568', gaia      , value=0.1)
    apply_sigma_based_on_catalog(Data, sigma, '568', ['t']     , value=0.2)
    # T09, T12, T14
    apply_sigma_based_on_catalog(Data, sigma, ['T09', 'T12', 'T14'], gaia, value=0.1)
    # 309
    apply_sigma_based_on_catalog(Data, sigma, '309', ['q', 't'], prg_code='&', value=0.3)
    apply_sigma_based_on_catalog(Data, sigma, '309', gaia      , prg_code='&', value=0.2)
    ### space telescopes
    sigma[np.isin(Data['Obs.Type'], ['s', 'S'])] = 1.5
    # HST
    sigma[Data['Obs.Code'] == '250'] = 0.05
    # Wise
    sigma[Data['Obs.Code'] == 'C51'] = 1.0
    ### Photographic
    i_photo = np.where(np.isin(Data['Obs.Type'], ['P', ' ', 'A', 'N']))[0]
    for i in i_photo:
        if Data['ET'][i]*cf.DAYS < spice.str2et('1890-01-01'):
            sigma[i] = 10.0
        elif Data['ET'][i]*cf.DAYS < spice.str2et('1950-01-01'):
            sigma[i] = 5.0
        else:
            sigma[i] = 2.5
    ### Other types (Table 5)
    Table5 = {'E': 0.2, 'H': 0.2, 'T': 0.5, 'e': 0.75, 'M': 2.0, 'n': 1.0}
    for otype, value in Table5.items():
        sigma[Data['Obs.Type'] == otype] = value
    ### assign uncertainties
    Data['sigma_RA'] = np.cos(Data['De']) * np.deg2rad(sigma/3600.)
    Data['sigma_De'] = np.deg2rad(sigma/3600.)
    return Data


### DIFFERENTIAL CORRECTION OF THE ORBIT --------------------------------------
# References: Farnocchia et al. (2015); Carpino et al. (2003)
def DiffCorr(Data, et0, x0, propagator, prop_args, max_iter):
    # >>> use SVD for normal equations and covariance calculation
    def solve_svd(N, d, tolerance=1e-12): 
        U, Sigma, _ = np.linalg.svd(N, hermitian=True)
        y = np.zeros_like(d)
        SigmaInv = np.zeros_like(Sigma)
        for i, sigma in enumerate(Sigma):
            if sigma > tolerance:
                y[i] = (U.T @ d)[i] / sigma
                SigmaInv[i] = 1. / sigma
        x = U @ y
        Cov = U @ np.diag(SigmaInv) @ U.T
        return x, Cov
    # <<<
    # parameters
    chi2_rec = 7.  # recovery threshold
    chi2_rej0 = 8.  # base rejection threshold
    alpha  = 0.25  # fraction of max chi-square to use as increased rejection threshold 
    frac   = 0.05  # maximum fraction of epochs discarded in a single step
    n, m = len(x0), len(Data['ET'])
    # initializations 
    x = x0
    flag = np.repeat(True, m) # epochs included in fit 
    Cov = np.zeros((n, n)) 
    m_rec, m_rej, m_use = 0, 0, m
    chi2 = np.zeros(m)
    ### begin main loop of differential correction
    print('Differential correction begins.')
    print('#iter.     RMS       chi-square    ||dx||_M    ||dx||_2      #rec   #rej   #use frac')
    RMS_hh = []
    for k in range(max_iter):
        # initialize normal equations and residuals
        BTWB = np.zeros((n, n))
        BTWz = np.zeros(n)
        ress = 0.
        # get residuals and partials for current initial state vector
        zm, Bm, ym = ResidualsAndPartials(Data, et0, x, propagator, prop_args)
        # go through all epochs and accumulate normal equations for those flagged for inclusion
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
                ress += z @ z * (206265.)**2 # for RMS in arc sec
                #ress += (z @ W @ z) # for RMS scaled to sigmas (actually cost func.)
                # covariance of post-fit residual if epoch i was included
                G = np.diag(sigma**2) - (B @ Cov @ B.T)
            else:
                # covariance of post-fit residual if epoch i was _not_ included
                G = np.diag(sigma**2) + (B @ Cov @ B.T)
            chi2[i] = z @ np.linalg.inv(G) @ z
        ## solve normal equations using SVD decomposition of normal matrix
        dx, Cov = solve_svd(BTWB, BTWz)
        # update parameters vector
        x = x - dx
        ## quality of fit statistics
        # RMS
        RMS = np.sqrt(ress/2./m_use)
        # running mean of RMS over last 5 iterations
        RMS_hh.append(RMS)
        if len(RMS_hh) >= 5:
            RMS_rm = np.mean(RMS_hh[-5:])
        else:
            RMS_rm = 1e10
        # norm wrt normal matrix (see Milani & Gronchi 2010; cf. Mahalanobis dist.)
        nrm = np.sqrt((dx @ BTWB @ dx)/n)
        # chi-square
        q = np.r_[zm[:,0], zm[:,1]]/np.r_[Data['sigma_RA'], Data['sigma_De']]
        chisq = np.sum(q[np.r_[flag, flag]]**2)
        ## screen output
        print('%4i   %12.6e %12.6e %12.6e %12.6e %6i %6i %6i %4.2f' % (k, RMS, chisq, nrm,
               np.linalg.norm(dx), m_rec, m_rej, m_use, m_use/m))
        # check for convergence to end loop:
        if (abs(RMS/RMS_rm-1.) < 1e-4) | (nrm < 1e-3) | (np.linalg.norm(dx) < 1e-10):
            break
        # >>> this section handles outliers rejection/recovery
        # adjust rejection threshld for this step:
        chi2_rej = max( chi2_rej0 + 400.*(1.2)**(-m_use), alpha * chi2[flag].max() ) 
        # mark epochs to be readmitted in the fit at next step...
        i_rec = np.where((flag == False) & (chi2 < chi2_rec))[0]
        # ... and update their flag
        flag[i_rec] = True
        m_rec = len(i_rec)
        # mark epochs to be excluded from fit at next step...
        i_mrk = np.where((flag == True) & (chi2 > chi2_rej))[0]
        # ...but only exclude up to (frac * m_use), in decreasing order of residual:
        m_rej = min(len(i_mrk), int(frac*m_use))
        chi2_mrk = np.flip(np.sort(chi2[i_mrk]))
        if len(i_mrk) > 0:
            k1 = [np.where(chi2 == chi2_m)[0][0] for chi2_m in chi2[i_mrk][:m_rej]]
            flag[k1] = False
        # update total number of epochs that will be used in the fit at next step:
        m_use = sum(flag)
        # <<<  
    print('End of differential correction.')
    # residuals of converged fit:
    res = np.r_[zm[:,0], zm[:,1]]
    # output as a dictionary
    Fit = {'x':x, 'Cov':Cov, 'res':res, 'flag':flag, 'chi-square':chisq, 'RMS':RMS, 
           'norm':nrm, 'ET0':et0, 'y':ym}
    return Fit

def ResidualsAndPartials(Data, et0, x, propagator, prop_args):
    n, m = len(x), len(Data['ET'])
    # call propagator function (its name is passed as input: "propagator"); returns
    # state vector y, state transition matrix PP, sensitivity matrix SS at all epochs
    # Note: propagator also takes care of light travel time iteration
    y, P, S = propagator(x, et0, Data['ET'], Data['RS'], prop_args)
    # initialize residuals and their partials matrix for current x
    B = np.zeros((m, 2, n))
    z = np.zeros((m, 2))
    # fill in B and z (Escobal 1976 formulae)
    for i in range(m):
        R_ = Data['RS'][i,:]
        r_, Pi, Si = y[i,:3], P[i,:,:], S[i,:,:]
        rho_ = r_ - R_
        rho = np.linalg.norm(rho_)
        cos_ra, sin_ra = np.cos(Data['RA'][i]), np.sin(Data['RA'][i])
        cos_de, sin_de = np.cos(Data['De'][i]), np.sin(Data['De'][i])
        A_ = np.array([-sin_ra, cos_ra, 0.])
        D_ = np.array([-sin_de*cos_ra, -sin_de*sin_ra, cos_de ])
        L_ = np.array([ cos_de*cos_ra,  cos_de*sin_ra, sin_de ])
        dL_ = (L_ - rho_/rho)
        # vector of residuals z
        z[i,:] = np.dot(dL_, A_), np.dot(dL_, D_)
        # design matrix ∂z/∂x
        B[i,:,:] = -np.c_[ np.r_[ A_, np.zeros(3) ]/rho @ np.block([Pi, Si]),
                           np.r_[ D_, np.zeros(3) ]/rho @ np.block([Pi, Si]) ].T
    return z, B, y



### TRAJECTORY PROPAGATION ----------------------------------------------------
# Two propagators are available, based on REBOUND/ASSIST and SciPy solve_ivp
# FIRST OPTION: use ASSIST (faster, probably more accurate, but less flexible)
def PropagateAssist(x, et0, et, RS, assist_params, n_tau_iter=2):
    ### iteration needed to account for light travel time:
    m = len(et)
    tau = np.zeros(m)
    for j in range(n_tau_iter):
        y, P, S = RunAssist(x, et0, et, tau, assist_params)
        tau = np.array([np.linalg.norm(y[i,:3]-RS[i,:])/cf.CC for i in range(m)])
    return y, P, S
        
def RunAssist(x, et0, et, tau, assist_params):
    # unpack parameters:
    forces, params_ng_radial = assist_params
    # set up REBOUND simulation with ASSIST extras
    def init_sim(t0, p0, ephem, params_ngforce):
        sim = rebound.Simulation()
        sim.add(p0)
        sim.t = t0
        extras = assist.Extras(sim, ephem)
        # forces to be included:
        extras.forces = forces
        # NG acceleration components:
        extras.particle_params = params_ngforce
        # parameters of radial dependence of NG force:
        extras.alpha = params_ng_radial['alpha']
        extras.r0    = params_ng_radial['r0']
        extras.nm    = params_ng_radial['m']
        extras.nn    = params_ng_radial['n']
        extras.nk    = params_ng_radial['k']
        return sim, extras
    t0 = et0
    t = et - tau
    p_  = x[6:]
    # set up ASSIST ephemerides extension
    ephem = assist.Ephem(cf.assist_planets_file, cf.assist_asteroids_file)
    # initial conditions of test particle
    p0_h = rebound.Particle(x=x[0], y=x[1], z=x[2], vx=x[3], vy=x[4], vz=x[5])
    # change heliocentric -> SSB frame (used by ASSIST)
    p0 = ephem.get_particle("sun", t0) + p0_h
    # parameters for non-gravitational forces
    nparms = len(p_)
    params_ngforce = np.zeros(3)
    for k in range(nparms):
        params_ngforce[k] = p_[k]
    # initialize simulation
    sim, extras = init_sim(t0, p0, ephem, params_ngforce)
    # prepare for integration 
    m = len(t)
    y = np.zeros((m,6))
    ### first integration with nominal values of p_
    for i, ti in enumerate(t):
        extras.integrate_or_interpolate(ti)
        sun = ephem.get_particle("sun", ti)
        y[i,0:3] = np.array(sim.particles[0].xyz) - np.array(sun.xyz)
        y[i,3:6] = np.array(sim.particles[0].vxyz) - np.array(sun.vxyz)
    ### set up integration with varying p_ components, to evaluate the sensitivity matrix
    eps = 1e-6
    S = np.zeros((m,6,nparms))
    y1 = np.zeros(6)
    for k in range(nparms):
        delta_params_ngforce = np.zeros(3)
        delta_params_ngforce[k] = params_ngforce[k] * eps
        # initialize simulation 
        sim, extras = init_sim(t0, p0, ephem, params_ngforce+delta_params_ngforce)         
        for i, ti in enumerate(t):
            extras.integrate_or_interpolate(ti)
            sun = ephem.get_particle("sun", ti)
            y1[0:3] = np.array(sim.particles[0].xyz) - np.array(sun.xyz)
            y1[3:6] = np.array(sim.particles[0].vxyz) - np.array(sun.vxyz)
            for j in range(6):
                S[i,j,k] = (y1[j] - y[i,j])/(delta_params_ngforce[k])
    ### evaluate state transition matrix using REBOUND variational particles
    # initialize simulation - note: STM simulation _must_ be run with NG forces set to zero! 
    sim, extras = init_sim(t0, p0, ephem, np.zeros(3))
    # add variational particles to calculate state transition matrix
    vp = [sim.add_variation(testparticle=0, order=1) for _ in range(6)]
    for j, axis in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz']):
        setattr(vp[j].particles[0], axis, 1)
    # prepare for integration 
    m = len(t)
    P = np.zeros((m,6,6))
    # run rebound simulation + assist
    for i, ti in enumerate(t):
        extras.integrate_or_interpolate(ti)
        # state transition matrix
        P[i] = np.array([np.r_[vp1.particles[0].xyz, vp1.particles[0].vxyz] for vp1 in vp]).T
    return y, P, S

# SECOND OPTION: integrate the equations of motion with the SciPy ode solver
# "solve_ivp"; slower option, but fully customizable equations (see "Derivs")
def PropagateSciPy(x, et0, et, RS, scipy_params, n_tau_iter=2):
    aNG, NGcoeff = scipy_params
    rtol = 1e-9
    atol = 1e-11
    #method = SWAG
    method = 'RK45'
    n = len(x)
    m = len(et)
    n_p = n-6  # number of parameters beyond initial state vector
    r0_, v0_, parms_ = x[0:3], x[3:6], x[6:]
    # form initial conditions for ODE integrator
    y0 = np.concatenate([r0_, v0_, np.eye(6).flatten(), np.zeros((6,n_p)).flatten()])
    # forward integration from et0 to et[-1]
    tspan_f = [et0, et[-1]]
    sol_f = solve_ivp(Derivs, tspan_f, y0, method=method, args=(parms_, aNG, NGcoeff),
            rtol=rtol, atol=atol, dense_output=True)
    # backward integration from et0 to et[0]
    tspan_b = [et0, et[0]]
    sol_b = solve_ivp(Derivs, tspan_b, y0, method=method, args=(parms_, aNG, NGcoeff),
            rtol=rtol, atol=atol, dense_output=True)
    ii_f = np.where(et >  et0)[0]
    ii_b = np.where(et <= et0)[0]
    ### iteration needed to account for light travel time:
    tau = np.zeros(m)
    for j in range(n_tau_iter):
        teval_b = et[ii_b]-tau[ii_b]
        teval_f = et[ii_f]-tau[ii_f]
        sol = np.array([sol_b.sol(tb).T for tb in teval_b] + [sol_f.sol(tf).T for tf in teval_f]) 
        tau = np.array([np.linalg.norm(sol[i,:3]-RS[i,:])/cf.CC for i in range(m)])
    # prepare output
    y = np.reshape(sol[:,  :6], (m, 6))
    P = np.reshape(sol[:,6:42], (m, 6, 6))
    S = np.reshape(sol[:,42: ], (m, 6, n_p))
    return y, P, S

# force model and linearization: see Montenbruck & Gill 2005, Chapters 3 and 7, resp.
def Derivs(t, y, parms_, aNG, NGcoeff):
    r_ = y[0:3]
    v_ = y[3:6]
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    ## >>> call non-gravitational acceleration function:
    if not aNG:
       aNG_, dadrNG, dadvNG, dadpNG = np.zeros(3), np.zeros((3,3)), np.zeros((3,3)), [] 
       n_p = 0
    else:
       aNG_, dadrNG, dadvNG, dadpNG = aNG(r_, v_, parms_, NGcoeff)
       n_p = len(parms_) 
    ## <<<
    ### acceleration
    a_ = -cf.MU_S*r_/r**3
    # add planets
    for i, TAG in enumerate(cf.TAG_P):
        s_ = spice.spkpos(TAG, t*cf.DAYS, 'J2000', 'NONE', '10')[0]/cf.AU
        a_ += - cf.MU_P[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # add asteroids
    for i, TAG in enumerate(cf.TAG_A):
        s_ = spice.spkpos(TAG, t*cf.DAYS, 'J2000', 'NONE', '10')[0]/cf.AU
        a_ += - cf.MU_A[i]*( (r_-s_)/np.linalg.norm(r_-s_)**3 + s_/np.linalg.norm(s_)**3 )
    # add GR correction
    a_ += (cf.MU_S/cf.CC**2/r**3)*( (4*cf.MU_S/r - v**2)*r_ + 4*np.dot(r_,v_)*v_ )
    # add non-gravitational term
    a_ = a_ + aNG_
    ### variational equations - note: asteroids, relativity omitted (GR would contribute to dadv)
    # 1. state transition matrix variation:
    PHI = np.reshape(y[6:42], (6, 6))
    ## dadr
    dadr = -cf.MU_S/r**3*( np.eye(3) - 3*np.outer(r_, r_)/r**2)
    # add planets
    for i, TAG in enumerate(cf.TAG_P):
        s_ = spice.spkpos(TAG, t*cf.DAYS, 'J2000', 'NONE', '10')[0]/cf.AU
        dadr += - cf.MU_P[i]/np.linalg.norm(r_-s_)**3 \
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



### OUTPUT --------------------------------------------------------------------
def SummaryPlot(object_name, Data, Fit, scaled=True):
    et, s_ra, s_de, RS = Data['ET'], Data['sigma_RA'], Data['sigma_De'], Data['RS']
    tt = Data['Obs.Epoch']
    x, z, flag, y = Fit['x'], Fit['res'], Fit['flag'], Fit['y'] 
    m = len(et)
    if scaled:
       res_ra, res_de, unit, ylim = z[:m] / s_ra, z[m:] / s_de, '($\\sigma$)', (-5, 5)     
    else:
       res_ra, res_de, unit, ylim = z[:m] * 206265., z[m:] * 206265., '(\")', (-10, 10)
    not_flag = np.logical_not(flag)
    dh = np.linalg.norm(y[:,:3], axis=1)
    rE, _ = spice.spkpos('399', et*cf.DAYS, 'J2000', 'NONE', '10')
    dg = np.linalg.norm(y[:,:3]-rE/cf.AU, axis=1)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ### make plot
    fig, axes = plt.subplot_mosaic('AAAX;BBBY;CCCZ', figsize=(7, 7), constrained_layout=True)
    # plot heliocentric and geocentric distance
    axes['A'].plot(tt, dh, '.', label='heliocentric')
    axes['A'].plot(tt, dg, 's', ms=3, label='geocentric')
    axes['A'].xaxis.set_ticklabels([])
    axes['A'].set_ylabel('Distance (AU)')
    axes['A'].set_ybound(lower=0.)
    axes['A'].legend()  
    # plot residuals in RA 
    axes['B'].axhspan(-1, 1, color='palegreen', alpha=0.5)
    axes['B'].plot(tt[flag], res_ra[flag], '.', color='#00356B', label='included')
    axes['B'].plot(tt[not_flag], res_ra[not_flag], 'x', ms=4, color='darkgray', label='excluded')
    axes['B'].axhline(color='orangered', lw=0.7)
    axes['B'].xaxis.set_ticklabels([])
    axes['B'].set_ylabel('R. A. Res. '+unit)
    axes['B'].set_ylim(ylim)
    axes['B'].legend()
    axes['B'].grid()
    # plot residuals in DE
    axes['C'].axhspan(-1, 1, color='palegreen', alpha=0.5)
    axes['C'].plot(tt[flag], res_de[flag], '.', color='#00356B', label='Decl.')
    axes['C'].plot(tt[not_flag], res_de[not_flag], 'x', ms=4, color='darkgray')
    axes['C'].axhline(color='orangered', lw=0.7)
    axes['C'].set_ylabel('Decl. Res. '+unit)
    axes['C'].set_xlabel('Date (UTC)')
    axes['C'].xaxis.set_major_locator(locator)
    axes['C'].xaxis.set_major_formatter(formatter)
    axes['C'].set_ylim(ylim)
    axes['C'].grid()
    #
    axes['X'].axis('off')
    axes['X'].text(-0.1, 0.9, 'Fit Epoch:')
    axes['X'].text(-0.1, 0.8, spice.et2utc(Fit['ET0']*cf.DAYS,'C', 0))
    axes['X'].text(-0.1, 0.65, 'Object:')
    axes['X'].text(-0.1, 0.55, object_name) 
    # add histograms
    axes['Y'].hist(res_ra[flag], bins=20, orientation='horizontal', color='#00356B')
    axes['Y'].axhline(color='orangered', lw=0.7)
    axes['Y'].set_ylim(ylim)
    axes['Y'].grid()
    axes['Z'].hist(res_de[flag], bins=20, orientation='horizontal', color='#00356B')
    axes['Z'].axhline(color='orangered', lw=0.7)
    axes['Z'].set_xlabel('Count')
    axes['Z'].set_ylim(ylim)
    axes['Z'].grid()
    plt.savefig('summary_plot.pdf')
    plt.show()
    plt.close()

def SummaryText(object_name, Data, Fit):
    n = len(Fit['x']) 
    xs1 = ['x ', 'y ', 'z ', 'vx', 'vy', 'vz', 'A1', 'A2', 'A3', 'DT']
    xs2 = np.r_[np.repeat('(au)', 3), np.repeat('(au/d)', 3), np.repeat('(10^-8 au/d^2)', 3),
    ['(days)']]
    ee1 = ['a', 'e', 'I', 'Ω', 'ω', 'M']
    ee2 = ['(au)', ' ', '(deg)', '(deg)', '(deg)', '(deg)']
    U = spice.pxform( 'J2000', 'ECLIPJ2000', Fit['ET0']*cf.DAYS )
    r_, v_ = U @ Fit['x'][0:3],   U @ Fit['x'][3:6]
    OE = spice.oscltx(np.r_[r_, v_], Fit['ET0'], cf.MU_S)
    oe = np.r_[ OE[9], OE[1], np.rad2deg(OE[2:6]) ] 
    A = JacobianTBP(r_, v_, cf.MU_S)[1]
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
        f.write('RMS = %8.3f (arc sec)\n' % Fit['RMS'])
        f.write('χ^2 = %8.2f\n' % Fit['chi-square'])
        f.write('No. residuals = %5i \n' % (2*sum(Fit['flag'])) )
        f.write('\n')
        f.write('# epochs = %5i \n' % sum(Fit['flag']))
        f.write('First epoch: '+spice.et2utc(Data['ET'][ 0]*cf.DAYS, 'C', 2)+'\n')
        f.write('Last  epoch: '+spice.et2utc(Data['ET'][-1]*cf.DAYS, 'C', 2)+'\n')
        f.write('\n')        
        f.write('Fit Epoch  : '+spice.et2utc(Fit['ET0']*cf.DAYS, 'C', 2)+'\n')
        f.write('\n')
        f.write('State vector (J2000 heliocentric frame)\n')
        for i in range(n):
            x_i = Fit['x'][i]
            sx_i = np.sqrt(Fit['Cov'][i,i])
            if i in [6, 7, 8]:
                x_i = x_i * 1e8
                sx_i = sx_i * 1e8
            f.write('%s = %15.9f ± %13.6e %s\n' % (xs1[i], x_i, sx_i, xs2[i]))
        f.write('\n')
        f.write('Orbital elements (ECLIPJ2000 heliocentric frame)\n')          
        for i in range(len(oe)):
            f.write('%s = %15.9f ± %13.6e %s\n' % (ee1[i], oe[i], sigma_oe[i], ee2[i]))
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
def InitOrbDet(Data, i1, i2, i3):
    tau1 = Data['ET'][i1]-Data['ET'][i2]
    tau3 = Data['ET'][i3]-Data['ET'][i2]
    Ri_ = Data['RS'][i1,:], Data['RS'][i2,:], Data['RS'][i3,:]
    e1_ = spice.radrec(1., Data['RA'][i1], Data['De'][i1])
    e2_ = spice.radrec(1., Data['RA'][i2], Data['De'][i2])
    e3_ = spice.radrec(1., Data['RA'][i3], Data['De'][i3])
    ei_ = e1_, e2_, e3_
    r2i = 3.0
    kmax = 50
    tol = 1e-6
    r2_, v2_, k = AnglesOnlyIOD(tau1, tau3, Ri_, ei_, cf.MU_S, r2i, kmax, tol)
    if k < kmax-1:
       print('Preliminary orbit determination converged in %i iterations' % k)
       exit_code = 0
    else:
       print('WARNING: Preliminary orbit determination did not converge in %i iterations' % kmax)
       exit_code = 1
    return r2_, v2_, exit_code


# Reference: Bate, Mueller, White (1971), Section 5.8, page 271
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
        #f1 = 1 - u2*tau1**2/2 + u2*p2*tau1**3/2 + u2*(u2 - 15*p2**2 + 3*q2)*tau1**4/24 \
        #       + u2*p2*(7*p2**2 - u2 - 3*q2)*tau1**5/8
        #g1 = tau1 - u2*tau1**3/6 + u2*p2*tau1**4/4 + u2*(u2 - 45*p2**2 + 9*q2)*tau1**5/120
        #f3 = 1 - u2*tau3**2/2 + u2*p2*tau3**3/2 + u2*(u2 - 15*p2**2 + 3*q2)*tau3**4/24 \
        #       + u2*p2*(7*p2**2 - u2 - 3*q2)*tau3**5/8
        #g3 = tau3 - u2*tau3**3/6 + u2*p2*tau3**4/4 + u2*(u2 - 45*p2**2 + 9*q2)*tau3**5/120
        f1, g1 = f_and_g(r2_, v2_, tau1, mu_s)
        f3, g3 = f_and_g(r2_, v2_, tau3, mu_s) 
        delta = (r2-r20)/r20
        #print(('%4i'+6*'%14.6e') % (k, r2, f1, f3, g1, g3, delta))
        if abs(delta) < tol:
            break
    return r2_, v2_, k
