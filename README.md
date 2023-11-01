# orbit_finder

Python orbit determination code. Works with astrometri data from the Minor 
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
* Input data: 1. MPC observatory codes and geodatum (see https://www.minorplanetcenter.net/iau/lists/ObsCodes.html)
              2. Input file following MPC format: the following examples are provided: example_1I.txt; example_469219.txt; example_523599.txt; example_6489.txt

## Description (TBD)
