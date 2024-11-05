# orbit_finder

Python orbit determination code. Works with astrometric data from the Minor Planet Center (MPC) database. 

## External dependencies
* NumPy 
* Matplotlib
* astroquery.jplhorizons: https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
* astropy-healpix: https://astropy-healpix.readthedocs.io/en/latest/
* SpiceyPy: https://spiceypy.readthedocs.io/en/stable/
* REBOUND: https://rebound.readthedocs.io/en/latest/ 
* ASSIST: https://assist.readthedocs.io/en/stable/

## Required input files
* For SpiceyPy: Meta-Kernel file "spice.mkn"
  (see https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem);
  Naif SPICE Kernels can be downloaded from: https://naif.jpl.nasa.gov/naif/data_generic.html
* For ASSIST: ephemerides files, such as "linux_p1550p2650.440", "sb441-n16.bsp" 
  (not part of this repository as they are relatively large in size; available from: https://assist.readthedocs.io/en/stable/installation/)
* For bias correction of astrometric data: the required file "bias.dat" can be downloaded from:  https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz, (see also Eggl et al. 2020)
* Input data: 
    1. MPC observatory codes and geodata "mpc_obs.txt" (cf. https://www.minorplanetcenter.net/iau/lists/ObsCodes.html)
    2. File with astrometric observations in MPC format; a few are included in this repository for testing and demonstration purposes.

## Description
The orbit determination procedure is described in detail in the arXiv preprint https://arxiv.org/abs/2304.06964, and references therein.
The current Python implementations improves the accuracy and extends the functionality of a previous MATLAB version.

The script contains functions for loading the data (assumed to be in MPC format), applying bias correction, performing preliminary orbit determination
on a subset of three user-specified epochs, and refining the initial guess by differential correction. A convenience functionality to query the state vector
at epoch from the JPL Horizons database is also included.

### Initial orbit determination
At the moment, an implementation of a Gaussian-like initial orbit determination method is available, which can be used to provide an initial guess for the
differential correction procedure. Alternatively, the state vector at a user-specified epoch obtained from a query to the JPL Horizons database can also be
used for the same purpose. The latter option is recommended for its higher accuracy and flexibility (e.g., it permits the user to freely choose the epoch 
at which the orbit will be determined).  

### Differential correction of the orbit
The differential correction procedure implements automatic outlier rejection (based on Carpino et al. 2003, Icarus, 166, 248), and can include the 
magnitude of the components of non-gravitational acceleration (prescribed according to the formulation of Marsden et al. 1973) in the solve-for vector
of parameters.

### Orbit propagation
Two propagators are available for the integration of the equations of motion and of the corresponding variational equations: 
1. Propagator based on REBOUND/ASSIST;
2. Propagator based on scipy.integrate.solve_ivp; a fully customizable implementation of the equations of motion is provided
   in this case; accepts external function implementing non-gravitational acceleration (see an example in non_grav_accel.py).
The scipy-version allows full control on the implementation of the equations of motion (add/remove force components; implement different parametrizations), 
but this freedom comes with a performance cost. 

## Usage
Some example use cases are showcased in fit_orbit.py. 

## Contributors
Federico Spada
