import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from load_data import LoadDataADES, DebiasData, AssignUncertaintiesVeres17
from initialize_orbit import QueryHorizons
from propagate_assist import PropagateWithAssist
from propagate_solivp import PropagateWithSolIVP, NonGravAccel
from extensisq import SWAG # use this or RK45, LSODA 
from propagate_heyoka import InitializeTaylorIntegrator, PropagateWithHeyoka
from differential_correction import DifferentialCorrection
from generate_output import SummaryText, SummaryPlot
import constants as cs


if __name__ == "__main__":

    # load SPICE Kernels
    spice.furnsh('_aux/spice_orbit_finder.mkn')

    # Auxiliary files for MPC data processing
    mpco_file = '_aux/obscodes_extended.json'
    ccod_file = '_aux/AstCatWithCodes.json'

    ### fit of 1I/'Oumuamua
    name = '1I'
    fit_epoch = '2017-11-23 TDB' 
    nga_parms = [1e-8]
    ###

    ### fit of 6489 Golevka
    #name = '6489'
    #fit_epoch = '2026-06-09 TDB'
    #nga_parms = []
    ###

 
    # load astrometry in ADES format (from MPC, .xml format)
    DataRaw, _ = LoadDataADES(
        name+'.xml', mpco_file, ccod_file, 
        defaultSigma=1.0,
        #start_date=''
        #end_date=''
    )
    # Apply catalog bias correction (Eggl+2020)
    Data1, _ = DebiasData(DataRaw, '_aux/bias.dat')
    # Apply Veres et al. (2017) uncertainties
    Data = AssignUncertaintiesVeres17(Data1)

    # ASSIST propagator initializations   
    forces = ['SUN', 'PLANETS', 'ASTEROIDS', 'NON_GRAVITATIONAL',
              'GR_EIH', 'EARTH_HARMONICS', 'SUN_HARMONICS']
    planets_eph_file = '_aux/linux_p1550p2650.440'
    asteroids_eph_file = '_aux/sb441-n16.bsp'
    invrsq = [1., 1., 2., 0., 0.] # 1/r^2 NGA
    ma73 = [0.1113, 2.808, 2.15, 5.093, 4.6142] # Marsden (1973) H2O NGA

    # choose propagator
    propagator = PropagateWithAssist
    prop_args  = forces, invrsq, planets_eph_file, asteroids_eph_file
    
    #propagator = PropagateWithSolIVP
    #prop_args = LSODA, invrsq, NonGravAccel # or RK45, or SWAG from extensisq

    #propagator = PropagateWithHeyoka
    #prop_args = InitializeTaylorIntegrator(ng_model=invrsq, ng_npars=len(nga_parms),
    #    vsop2013_thresh=1e-6, elp2000_thresh=1e-4)[1]

    # initialize orbit fit
    x00 = QueryHorizons(name, fit_epoch)
    x0 = np.hstack((x00, nga_parms))

    # perform orbit fit
    Fit = DifferentialCorrection(
        Data, fit_epoch, x0, propagator, prop_args,
        max_iter=15, chi2_rec=7., chi2_rej=15.,
    )

    # generate output
    SummaryPlot(
        name, Data, Fit, scaled=True, nbins=41, ylim=(-5,5), fname='summary_plot.pdf')
    SummaryText(name, Data, Fit, fname='summary_fit.txt')

    # reprint to screen for convenience
    with open('summary_fit.txt', 'r') as f:
        print(f.read())

    # unload SPICE Kernels
    spice.kclear()

