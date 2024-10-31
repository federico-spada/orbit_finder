import numpy as np
import spiceypy as spice
from orbit_finder import *
from non_grav_accel import NonGravAccel
from astroquery.jplhorizons import Horizons

spice.furnsh('spice.mkn')


### initializations
object_name = '523599'
fit_epoch = '2018-09-13.0'
parms0_ = np.array([1e-12, 1e-12, 1e-12])
#parms0_ = []
propagator = PropagateAssist
prop_args  = cf.all_forces 
#propagator = PropagateSciPy
#prop_args = NonGravAccel

### Load data
# file with observatory data
fobss = 'mpc_obs.txt'
# file with observations
fdata = object_name+'.txt'
et, ra, de, s_ra, s_de, RS, JD, OC = LoadDataMPC(fobss, fdata)
et0 = spice.str2et(fit_epoch)/cf.days
title = 'aaa'
max_iter = 25

# Orbit initialization
# load JPL Horizons state vector as initial guess:
target = object_name #obj_name.split('_')[-2] + ' ' + obj_name.split('_')[-1]
epoch  = et0 + 2451545.0
query = Horizons(target, location='@10', epochs=epoch)
vec = query.vectors(refplane='earth')
x0 = np.array([vec['x'][0] , vec['y'][0 ], vec['z'][0],
               vec['vx'][0], vec['vy'][0], vec['vz'][0]])
# initial guess for state vector
x0 = np.r_[x0, parms0_]

###  differential correction
x, Cov, RMS, res, flag = DiffCorr(et, ra, de, s_ra, s_de, RS, et0, x0, 
                                  propagator, prop_args, max_iter)

SummaryPlot(et, res, s_ra, s_de, RS, et0, propagator, prop_args, flag, x, scaled=True)

spice.kclear()

