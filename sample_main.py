import numpy as np
import spiceypy as spice
from orbit_finder import *
from non_grav_accel import NonGravAccel


def RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,prop_args,max_iter,label):
    title = (obj_name+label).replace('C_','C/').replace('_',' ')
    print(title)
    #load data up to perihelion 
    fobss = 'mpc_obs.txt'
    fdata = obj_name+'.txt'
    et,ra,de,s_ra,s_de,RS,JD,OC = LoadDataMPC(fobss,fdata,start_date=start_date,end_date=end_date)
    # preliminary orbit determination  
    i1, i2, i3 = i_iod
    r2_, v2_, _ = InitOrbDet(et,ra,de,s_ra,s_de,RS,i1,i2,i3)
    et0 = et[i2]
    # differential correction, PRE, GR
    x0 = np.r_[r2_,v2_, parms_]
    x,Cov,z,chi2,B,flag,u,X2 = DiffCorr(et,ra,de,s_ra,s_de,RS,et0,x0,propagator,prop_args,max_iter)
    # output, PRE, GR
    n_fit = len(et[flag])
    s_x = np.sqrt(np.diagonal(Cov))
    s = np.r_[s_ra, s_de]
    mask = np.r_[flag, flag]
    RMS = np.sqrt(sum((z[mask]/s[mask])**2)/len(z))
    print('RMS = %8.3f, n_epochs = %4i' % (RMS, n_fit))
    if len(x) > 6:
        print('A1 = %8.3f ± %8.3f  [10^-8 au/d^2]' % (x[6]*1e8, s_x[6]*1e8))
    if len(x) > 7:
        print('A2 = %8.3f ± %8.3f  [10^-8 au/d^2]' % (x[7]*1e8, s_x[7]*1e8))
    if len(x) > 8:
        print('A3 = %8.3f ± %8.3f  [10^-8 au/d^2]' % (x[8]*1e8, s_x[8]*1e8))
    fname = obj_name+label+'.pdf'
    plt.figure()
    PlotResiduals(et,z,s_ra,s_de,RS,et0,propagator,prop_args,flag,x,title)
    #plt.savefig(fname)
    plt.show()
    plt.figure()
    fname = obj_name+label+'_hist.pdf'    
    PlotHistRes(z,s_ra,s_de,flag,title)
    #plt.savefig(fname)
    plt.show()
    print(' ')
    return RMS, x, s_x, n_fit



if __name__ == '__main__':

    ### provide all spice Kernels via meta-Kernel
    spice.furnsh('spice.mkn')

    ### 
    ### Fit only the pre-perihelion arc of comet C/1998 P1; ASSIST propagator, with NG acceleration
    ###
    # name of the file with astrometric data
    obj_name = 'C_1998_P1'
    # start_date and end_date define the range of epochs to be used 
    start_date = None
    end_date = '1998-12-01'
    # indices of epochs to be used for initial orbit determination
    i_iod = 50, 70, 90
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and three, leave empty for purely gravitational fit
    parms_ = np.array([1e-12,1e-12,1e-12])
    # maximum number of iteration in the differential correction procedure
    max_iter = 9
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateAssist
    # additional parameters for propagator (for assist, it's the list of forces to be included)
    prop_args = all_forces
    # string to be appended to names of files with plot output 
    label ='_PRE_A1A2A3'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)


    ### 
    ### Same as above, but with SciPy propagator (note: the NG acceleration implementation differs!)
    ###
    # name of the file with astrometric data
    obj_name = 'C_1998_P1'                   
    # start_date and end_date define the range of epochs to be used 
    start_date = None                        
    end_date = '1998-12-01'       
    # indices of epochs to be used for initial orbit determination
    i_iod = 50, 70, 90
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and four, leave empty for purely gravitational fit
    parms_ = np.array([1e-12,1e-12,1e-12])   
    # maximum number of iteration in the differential correction procedure
    max_iter = 9
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateSciPy
    # additional parameters for propagator (for scipy, it's the function with the NG acceleration)
    prop_args = NonGravAccel
    # string to be appended to names of files with plot output 
    label ='_PRE_NG_A1A2A3'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)
    

    ### 
    ### 6489 Golevka, purely gravitational fit, with ASSIST
    ###
    # name of the file with astrometric data
    obj_name = '6489'
    # start_date and end_date define the range of epochs to be used 
    start_date = None
    end_date   = None
    # indices of epochs to be used for initial orbit determination
    i_iod = 858, 866, 873
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and four, leave empty for purely gravitational fit
    parms_ = np.array([])
    # maximum number of iteration in the differential correction procedure
    max_iter = 15
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateAssist
    # additional parameters for propagator 
    prop_args = all_forces
    # string to be appended to names of files with plot output 
    label ='_FULL_GR'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)


    ### 
    ### 'Oumuamua, gravity only, with ASSIST
    ###
    # name of the file with astrometric data
    obj_name = '1I'
    # start_date and end_date define the range of epochs to be used 
    start_date = None
    end_date   = None
    # indices of epochs to be used for initial orbit determination
    i_iod = 5, 15, 30
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and four, leave empty for purely gravitational fit
    parms_ = np.array([])
    # maximum number of iteration in the differential correction procedure
    max_iter = 3
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateAssist
    # additional parameters for propagator 
    prop_args = all_forces
    # string to be appended to names of files with plot output 
    label ='_GR'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)


    ### 
    ### 'Oumuamua, radial-only NG acceleration, with ASSIST
    ###
    # name of the file with astrometric data
    obj_name = '1I'
    # start_date and end_date define the range of epochs to be used 
    start_date = None
    end_date   = None
    # indices of epochs to be used for initial orbit determination
    i_iod = 5, 15, 30
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and four, leave empty for purely gravitational fit
    parms_ = np.array([1e-12])
    # maximum number of iteration in the differential correction procedure
    max_iter = 15
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateAssist
    # additional parameters for propagator 
    prop_args = all_forces
    # string to be appended to names of files with plot output 
    label ='_NG_A1'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)


    ### 
    ### RM 2003, radial-only NG acceleration, with ASSIST
    ###
    # name of the file with astrometric data
    obj_name = '523599'
    # start_date and end_date define the range of epochs to be used 
    start_date = None
    end_date   = None
    # indices of epochs to be used for initial orbit determination
    i_iod = 10, 30, 70
    # initial values of the parameters of the non-gravitational force
    # for assist version, specify between one and four, leave empty for purely gravitational fit
    parms_ = np.array([1e-12,1e-12])
    # maximum number of iteration in the differential correction procedure
    max_iter = 15
    # choose propagator (ASSIST or SciPy)
    propagator = PropagateAssist
    # additional parameters for propagator 
    prop_args = all_forces
    # string to be appended to names of files with plot output 
    label ='_NG_A1A2A3'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)

    ### release the spice Kernels 
    spice.kclear()
