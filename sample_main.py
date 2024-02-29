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
    fname = obj_name+label+'.pdf'
    PlotResiduals(et,z,s_ra,s_de,RS,et0,propagator,prop_args,flag,x,title,fname)
    plt.clf()
    fname = obj_name+label+'_hist.pdf'    
    PlotHistRes(z,s_ra,s_de,flag,title,fname)
    print(' ')
    return RMS, x, s_x, n_fit



if __name__ == '__main__':

    ### provide all spice Kernels via meta-Kernel
    spice.furnsh('spice.mkn')

    # -----------------------------------------------------------------------------------
    # object
    obj_name = 'C_1998_P1'

    ### FULL arc fit
    # GR
    start_date = None
    end_date = '1998-12-01'
    i_iod = 50, 70, 90
    parms_ = np.array([])
    max_iter = 9
    propagator = PropagateSciPy
    prop_args = NonGravAccel
    label ='_FULL_GR'
    RMS, x, s_x, n_fit = RunFit(obj_name,start_date,end_date,i_iod,parms_,propagator,
    prop_args,max_iter,label)


    ### release the spice Kernels 
    spice.kclear()
