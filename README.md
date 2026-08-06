# orbit_finder

`orbit_finder` is a Python package for orbit determination of small Solar System bodies from astrometric observations. It is designed to work primarily with observations retrieved from the Minor Planet Center (MPC) database and provides tools for data preprocessing, initial orbit determination, differential correction, and high-precision orbit propagation.

The current Python implementation evolved from an earlier MATLAB version.

## Dependencies

The code requires the following Python packages:

- NumPy
- Matplotlib
- [astroquery.jplhorizons](https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html)
- [astropy-healpix](https://astropy-healpix.readthedocs.io/en/latest/)
- [SpiceyPy](https://spiceypy.readthedocs.io/en/stable/)
- [REBOUND](https://rebound.readthedocs.io/en/latest/)
- [ASSIST](https://assist.readthedocs.io/en/stable/)
- [heyoka.py](https://bluescarni.github.io/heyoka.py/index.html)

## Required external data files

Some external data files are required to run the full functionality of the code. Due to their size and/or external licensing, they are not included in this repository.

### SPICE kernels

For SpiceyPy-based propagation, a SPICE meta-kernel file (e.g. `spice_orbit_finder.mkn`) is required. This file should point to the appropriate SPICE kernels installed on the local system.

See the [SpiceyPy kernel management documentation](https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem).

NAIF SPICE kernels can be downloaded from:
https://naif.jpl.nasa.gov/naif/data_generic.html

### ASSIST ephemerides

The ASSIST propagator requires planetary ephemerides files:

- `linux_p1550p2650.440` or `linux_m13000p17000.441`
- `sb441-n16.bsp`

These files are not included in this repository because of their size. They are available from the [ASSIST installation documentation](https://assist.readthedocs.io/en/stable/installation/).

### Astrometric bias correction

Bias correction of astrometric observations requires the file:

- `bias.dat`

The file can be obtained from:
https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz

See also Eggl et al. (2020).

### Auxiliary MPC data files

The following auxiliary files are required for processing MPC observations in ADES format:

- `obscodes_extended.json`
- `AstCatWithCodes.json`

### Input observations

Input observations should be provided in MPC ADES XML format (`.xml`). A few example files are included in this repository for testing and demonstration purposes.

## Description

The orbit determination procedure implemented in this package is described in detail in:

- https://arxiv.org/abs/2304.06964
- https://arxiv.org/abs/2603.00782

and in the references therein.

The code includes routines for:

- loading and preprocessing astrometric observations;
- applying astrometric bias corrections;
- preliminary orbit determination;
- differential correction and parameter estimation;
- orbit propagation and uncertainty propagation.

## Initial orbit determination

The current implementation provides a Gaussian-like initial orbit determination method based on observations at three user-selected epochs. This method provides an initial state vector for the subsequent differential correction procedure.

Alternatively, an initial state vector can be obtained from the JPL Horizons database and used as the starting point for orbit determination. This option is generally recommended because it provides higher accuracy and allows the user to freely choose the reference epoch.

## Differential correction

The differential correction procedure includes:

- iterative least-squares orbit refinement;
- automatic outlier rejection based on Carpino et al. (2003), *Icarus*, 166, 248;
- optional estimation of non-gravitational acceleration parameters using the Marsden et al. (1973) formulation.

## Orbit propagation

Three propagators are currently available:

1. **REBOUND/ASSIST propagator**

   Uses numerical planetary ephemerides (JPL DE440/DE441) and is suitable for high-accuracy orbit propagation.

2. **`scipy.integrate.solve_ivp` propagator**

   A fully customizable implementation where the user can specify the equations of motion and optional non-gravitational accelerations.

3. **heyoka.py propagator**

   Provides high speed and high numerical accuracy. It should be used with caution because planetary perturbations are obtained from analytical models rather than numerical ephemerides.

## Usage

Example workflows are provided in the driver script: `orbit_finder.py`

## Contributors

Federico Spada

