# orbit_finder

Python orbit determination code. Works with astrometric data from the Minor Planet Center (MPC) database. 

## External dependencies
* NumPy 
* Matplotlib
* SciPy
* astroquery.jplhorizons (part of Astropy): https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
* SpiceyPy: https://spiceypy.readthedocs.io/en/stable/
* extensisq: https://github.com/WRKampi/extensisq

## Required input files
* For SpiceyPy: Meta-Kernel file "spice.mkn"
  (see https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem)
* Input data: 
    1. MPC observatory codes and geodatum "mpc_obs.txt" (cf. https://www.minorplanetcenter.net/iau/lists/ObsCodes.html)
    2. File with astrometric observations in MPC format; the following are provided for testing purposes: 1I.txt; 469219.txt; 523599.txt; 6489.txt

## Description
The orbit determination procedure is described at length in https://arxiv.org/abs/2304.06964, and in the references
given there, together with a previous version of the code, written in MATLAB. 
This Python rewriting improves the accuracy, and extends the functionality of the MATLAB version.

The script contains functions for loading the data (in MPC format), performing preliminary orbit determination
on a subset of three user-specified epochs, and refining the initial guess by differential correction. 
The differential correction procedure implements automatic outlier rejection (based on Carpino et al. 
2003, Icarus, 166, 248), and solves for the magnitude of the components of the non-gravitational acceleration 
(if requested by the user).

Note: in this version, the equations of motion and the corresponding variational equations are solved using the initial
value problem solver "solve_ivp", which is part of ScyPy (in contrast to the main branch version, where the numerical
integration of the trajectory is performed with REBOUND/ASSIST); in general, the results are comparable, although the
speed performance of ASSIST (main branch) is superior. In this version, however, the user has full control on the 
implementation of the equations of motion. This feature is useful, for instance, to add/remove a specific perturber 
(ASSIST only allows to remove the PLANETS and ASTEROIDS in bulk), or to modify the parametrization of the non-gravitational
acceleration (change basis vector decomposition, change g(r), scalings, etc.: these are all hard-coded and non-modifiable
in the current version of ASSIST). In summary, this is a better version for tinkering and experimenting, the extra freedom
coming at a (usually reasonable) cost in performance.

## Usage
Some example use cases are provided and run through the RunFit function. A detailed description of the input for
the use cases is as follows:
* obj_name: string with the name of the object whose orbit is to be fit; a .txt file with matching name containing
  astrometric data (in MPC format) should be present in the working directory
* i1, i2, i3: integers, specifying the indices of the three epochs to be used in the preliminary orbit determination;
  in general, the choice of these three epochs is up to the user, and it is one of the few cases where human intervention
  and case-by-case judgment is required
* parm_: numpy array, should contain one/three elements as initial guess for the components of the non-gravitational
  acceleration; three components are required for "ng_acc"="RTN" or "ng_acc=ACN", one component for "ng_acc=radial" or
  "ng_acc=tangential", not used if "ng_acc=none" or not set
* ng_acc, string, implementing different decomposition of the non gravitational acceleration; "RTN": (radial, tangential, normal),
  "ACN": (along-track, cross-track, normal), "radial", "tangential": self-explanatory, "none" or not assigned: no non-gravitational
  acceleration will be considered in the differential correction procredure

## Contributors
Federico Spada
