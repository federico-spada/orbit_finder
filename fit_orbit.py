import numpy as np
import spiceypy as spice
from orbit_finder import *
from non_grav_accel import NonGravAccel
from astroquery.jplhorizons import Horizons

spice.furnsh('spice.mkn')

forces = ['SUN', 'PLANETS', 'ASTEROIDS', 'NON_GRAVITATIONAL',
          'EARTH_HARMONICS', 'SUN_HARMONICS', 'GR_EIH']
ma73_h2o = {'alpha':0.1113, 'r0':2.808, 'm':2.15, 'n':5.093, 'k':4.6142}
inv_rsq  = {'alpha':1., 'r0':1, 'm':2., 'n':5.093, 'k':0.}

# A few use-cases

# (6489) Golevka 
object_name = '6489'
fit_epoch = '2021-01-21.0'
parms0_ = []
propagator = PropagateAssist
prop_args  = forces, inv_rsq

# (523599) 2003 RM 
#object_name = '523599'
#fit_epoch = '2017-05-05.0'
#parms0_ = np.array([1e-13, 1e-13])
#propagator = PropagateAssist
#prop_args  = forces, inv_rsq

# 1I/'Oumuamua 
#object_name = '1I'
#fit_epoch = '2018-01-01.0'
#parms0_ = np.array([1e-10])
#propagator = PropagateAssist
#prop_args  = forces, inv_rsq
#propagator = PropagateSciPy
#prop_args = NonGravAccel, inv_rsq

# C/1998 P1
#object_name = 'C_1998_P1'
#fit_epoch = '1998-11-03'
#parms0_ = np.array([1e-7, 1e-7, 1e-7, 10.])
#propagator = PropagateSciPy
#prop_args = NonGravAccel, ma73_h2o
#propagator = PropagateAssist
#prop_args = forces, ma73_h2o

start_date = None
end_date = None


### Load data
fobss = 'mpc_obs.txt'
fdata = object_name+'.txt'
Data = LoadDataMPC(fobss, fdata, start_date=start_date, end_date=end_date)
et0 = spice.str2et(fit_epoch)/cf.DAYS
max_iter = 25

fbias = 'bias.dat'
Data = DebiasData(fbias, Data)

### Orbit fit initialization
if '_' in object_name:
    target = object_name.split('_')[-2] + ' ' + object_name.split('_')[-1]
else:
    target = object_name 
epoch  = et0 + spice.j2000() # TDB, _not_ UTC!
query = Horizons(target, location='@10', epochs=epoch)
vec = query.vectors(refplane='earth')
xH = np.array([vec[key][0] for key in ['x', 'y', 'z', 'vx', 'vy', 'vz']])
x0 = np.r_[xH, parms0_]

### Differential correction
Fit = DiffCorr(Data, et0, x0, propagator, prop_args, max_iter)

### Output
SummaryPlot(object_name, Data, Fit, scaled=True)
SummaryText(object_name, Data, Fit)

spice.kclear()

