import numpy as np
import spiceypy as spice
from orbit_finder import *
from non_grav_accel import NonGravAccel
from astroquery.jplhorizons import Horizons

spice.furnsh('spice.mkn')

# A few use-cases

# (6489) Golevka 
object_name = '6489'
fit_epoch = '2022-01-21.0'
parms0_ = []
propagator = PropagateAssist
prop_args  = cf.all_forces 

# (523599) 2003 RM 
#object_name = '523599'
#fit_epoch = '2023-09-13.0'
#parms0_ = np.array([1e-11, 1e-11, 1e-11])
#propagator = PropagateAssist
#prop_args  = cf.all_forces

# 1I/'Oumuamua 
#object_name = '1I'
#fit_epoch = '2018-01-01.0'
#parms0_ = np.array([1e-10]) # purely radial!
#propagator = PropagateAssist
#prop_args  = cf.all_forces 

# C/1998 P1
#object_name = 'C_1998_P1'
#fit_epoch = '1998-05-31'
#parms0_ = np.array([1e-10, 1e-10, 1e-10])
#propagator = PropagateSciPy
#prop_args = NonGravAccel


### Load data
fobss = 'mpc_obs.txt'
fdata = object_name+'.txt'
Data = LoadDataMPC(fobss, fdata)
et0 = spice.str2et(fit_epoch)/cf.days
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

