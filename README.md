# orbit_finder

Python orbit determination code. Works with astrometric data from the Minor Planet Center (MPC) database. 

## External dependencies
* NumPy 
* Matplotlib
* seaborn
* astroquery.jplhorizons (part of Astropy): https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
* SpiceyPy: https://spiceypy.readthedocs.io/en/stable/
* REBOUND: https://rebound.readthedocs.io/en/latest/ 
* ASSIST: https://assist.readthedocs.io/en/stable/

## Required input files
* For SpiceyPy: Meta-Kernel file "spice.mkn"
  (see https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem);
  Naif SPICE Kernels can be downloaded from: https://naif.jpl.nasa.gov/naif/data_generic.html
* For ASSIST: ephemerides files "linux_p1550p2650.440", "sb441-n16.bsp" 
  (not part of this repository; available from: https://assist.readthedocs.io/en/stable/installation/)
* Input data: 
    1. MPC observatory codes and geodatum "mpc_obs.txt" (cf. https://www.minorplanetcenter.net/iau/lists/ObsCodes.html)
    2. File with astrometric observations in MPC format; some are provided for testing and demonstration purposes.

## Description
The orbit determination procedure is described at length in https://arxiv.org/abs/2304.06964, and the references given there,
together with a previous version of the code, written in MATLAB. 
This Python rewriting improves the accuracy, and extends the functionality of the MATLAB version.

The script contains functions for loading the data (in MPC format), performing preliminary orbit determination
on a subset of three user-specified epochs, and refining the initial guess by differential correction. 
The differential correction procedure implements automatic outlier rejection (based on Carpino et al. 
2003, Icarus, 166, 248), and solves for the magnitude of the components of the non-gravitational acceleration 
(if requested by the user).

Two propagators are available for the integration of the equations of motion and of the corresponding variational equations: 
1. Propagator based on REBOUND/ASSIST;
2. Propagator based on scipy.integrate.solve_ivp; a fully customizable implementation of the equations of motion is provided
   in this case; accepts external function implementing non-gravitational acceleration (see an example in non_grav_accel.py).
The scipy-version allows full control on the implementation of the equations of motion (terms added/removed, parametrizations), 
but this freedom comes with a performance cost. 

## Usage
Some example use cases are detailed in sample_main.py. 

## Contributors
Federico Spada
