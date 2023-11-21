# orbit_finder

Python orbit determination code. Works with astrometric data from the Minor Planet Center (MPC) database. 

## External dependencies
* NumPy 
* Matplotlib
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
    2. File with astrometric observations in MPC format; the following are provided for testing purposes: 1I.txt; 469219.txt; 523599.txt; 6489.txt

## Description
The orbit determination procedure is described at length in https://arxiv.org/abs/2304.06964, and the references given there,
together with a previous version of the code, written in MATLAB. 
This Python rewriting improves the accuracy, and extends the functionality of the MATLAB version.

The script contains functions for loading the data (in MPC format), performing preliminary orbit determination
on a subset of three user-specified epochs, and refining the initial guess by differential correction. 
The differential correction procedure implements automatic outlier rejection (based on Carpino et al. 
2003, Icarus, 166, 248), and solves for the magnitude of the components of the non-gravitational acceleration 
(if requested by the user).

This is the main branch, in which the integration of the equations of motion and of the corresponding variational equations 
is performed using REBOUND/ASSIST. A separate branch in which the numerical integration of the trajectory is accomplished
using the initial value problem solver "solve_ivp" (part of ScyPy), is available (check out "scipy-version"). The
scipy-version allows full control on the implementation of the equations of motion (terms added/removed, parametrizations), 
but this freedom comes with a performance cost. 

## Usage
Some example use cases are provided and run through the RunFit function. 
A detailed description of the input for the use cases is as follows:
* obj_name: string with the name of the object whose orbit is to be fit; a .txt file with matching name 
  containing astrometric data (in MPC format) should be present in the working directory
* i1, i2, i3: integers, specifying the indices of the three epochs to be used in the preliminary orbit determination; 
  in general, the choice of these three epochs is up to the user, and it is one of the few cases where human 
  intervention and case-by-case judgment is required
* parm_: numpy array, should contain up to three elements as initial guess for the radial, tangential, normal
  components, respectively, of the non-gravitational acceleration (see ASSIST docs for details); set to empty
  numpy array to switch off the non-gravitational acceleration in the differential correction
* forces: list of strings, useful to interact with the ASSIST capability to switch on/off certain force components
  in the equations of motion

## Contributors
Federico Spada
