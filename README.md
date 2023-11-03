# orbit_finder

Python orbit determination code. Works with astrometric data from the Minor 
Planet Center (MPC) database. 

## External dependencies
* NumPy 
* Matplotlib
* astroquery.jplhorizons (part of Astropy): https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
* SpiceyPy: https://spiceypy.readthedocs.io/en/stable/
* REBOUND: https://rebound.readthedocs.io/en/latest/ 
* ASSIST: https://assist.readthedocs.io/en/stable/

## Required input files
* For SpiceyPy: Meta-Kernel file "spice.mkn"
  (see https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem)
* For ASSIST: ephemerides files "linux_p1550p2650.440", "sb441-n16.bsp" 
  (not part of this repository; available from: https://assist.readthedocs.io/en/stable/installation/)
* Input data: 
    1. MPC observatory codes and geodatum "mpc_obs.txt" (cf. https://www.minorplanetcenter.net/iau/lists/ObsCodes.html)
    2. File with astrometric observations in MPC format; the following examples are provided: example_1I.txt; example_469219.txt; example_523599.txt; example_6489.txt

## Description
The orbit determination procedure is described at length in https://arxiv.org/abs/2304.06964,
together with a previous version of the code, written in MATLAB.
The script contains functions for loading the data (in MPC format), performing preliminary orbit determination
on a subset of three user-specified epochs, and refining the initial guess by differential correction of the
orbit. The differential correction procedure implements automatic outlier rejection (based on Carpino et al. 
2003, Icarus, 166, 248), and solves for the magnitude of components of the non-gravitational acceleration (if
requested by the user).

## Usage
Lines 486-507 of the script contain the basic setup, and require manual intervention for a user-defined run.

## Contributors
Federico Spada
