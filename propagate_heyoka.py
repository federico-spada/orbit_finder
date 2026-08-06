import numpy as np
import heyoka as hy

import constants as cs

mu_elp2000_scale_factor = (cs.DAYS)**2 / (cs.AU*1e3)**3
CC = cs.CC * cs.DAYS / cs.AU

def InitializeTaylorIntegrator(ng_model, ng_npars, vsop2013_thresh=1e-8, elp2000_thresh=1e-6):
    print(vsop2013_thresh, elp2000_thresh)
    # load gravitational parameters (VSOP2013):
    # Sun, Mercury, Venus, Earth+Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
    mu = hy.model.get_vsop2013_mus() # au^3/d^2
    # gravitational parameters of Earth, Moon, and their mass ratio (ELP2000):
    mu_e = hy.model.get_elp2000_mus()[0] * mu_elp2000_scale_factor # au^3/d^2
    mu_m = hy.model.get_elp2000_mus()[1] * mu_elp2000_scale_factor
    em_ratio = mu_e / mu_m
    # define variables
    x, y, z, vx, vy, vz = hy.make_vars('x', 'y', 'z', 'vx', 'vy', 'vz')
    r = hy.sqrt(x**2 + y**2 + z**2)
    inv_r3 = r ** (-3.0)
    # solar gravity
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = - mu[0] * x * inv_r3 
    dvydt = - mu[0] * y * inv_r3 
    dvzdt = - mu[0] * z * inv_r3
    # add planets (ICRF positions from VSOP2013)
    id_planets = [i for i in range(1, 10) if i != 3]
    for i in id_planets:
        sx, sy, sz = np.array(
            hy.model.vsop2013_cartesian_icrf(i, (hy.time/cs.DMIL), thresh=vsop2013_thresh)[:3]
        )
        inv_s3 = (sx**2 + sy**2 +sz**2) ** (-3/2.0)
        inv_d3 = ((x-sx)**2 + (y-sy)**2 + (z-sz)**2) ** (-3/2.0)
        dvxdt += mu[i] * ( (sx - x) * inv_d3 - sx * inv_s3 )
        dvydt += mu[i] * ( (sy - y) * inv_d3 - sy * inv_s3 )
        dvzdt += mu[i] * ( (sz - z) * inv_d3 - sz * inv_s3 )
    # >>> special case: Earth-Moon system:
    # ICRF position of Earth-Moon barycenter from VSOP2013:
    xEMB, yEMB, zEMB = np.array(
        hy.model.vsop2013_cartesian_icrf(3, (hy.time/cs.DMIL), thresh=vsop2013_thresh)[:3]
    )
    # geocentric position of the Moon from ELP2000 (converted: FK5 -> ICRS; km -> au)
    xm, ym, zm = (
        np.array(
            hy.model.rot_fk5j2000_icrs(
                hy.model.elp2000_cartesian_fk5((hy.time/cs.DCTY), thresh=elp2000_thresh)
            )
        ) / cs.AU
    ) 
    # Earth
    sx = xEMB - xm / (1.0 + em_ratio)
    sy = yEMB - ym / (1.0 + em_ratio)
    sz = zEMB - zm / (1.0 + em_ratio)
    inv_s3 = (sx**2 + sy**2 +sz**2) ** (-3/2.0)
    inv_d3 = ((x-sx)**2 + (y-sy)**2 + (z-sz)**2) ** (-3/2.0)
    dvxdt += mu_e * ( (sx - x) * inv_d3 - sx * inv_s3 )
    dvydt += mu_e * ( (sy - y) * inv_d3 - sy * inv_s3 )
    dvzdt += mu_e * ( (sz - z) * inv_d3 - sz * inv_s3 )
    # Moon
    sx = xEMB + xm * em_ratio / (1.0 + em_ratio)
    sy = yEMB + ym * em_ratio / (1.0 + em_ratio)
    sz = zEMB + zm * em_ratio / (1.0 + em_ratio)
    inv_s3 = (sx**2 + sy**2 +sz**2) ** (-3/2.0)
    inv_d3 = ((x-sx)**2 + (y-sy)**2 + (z-sz)**2) ** (-3/2.0)
    dvxdt += mu_m * ( (sx - x) * inv_d3 - sx * inv_s3 )
    dvydt += mu_m * ( (sy - y) * inv_d3 - sy * inv_s3 )
    dvzdt += mu_m * ( (sz - z) * inv_d3 - sz * inv_s3 )
    # <<<
    # >>> add GR term
    r_dot_v = x * vx + y * vy + z * vz
    v2 = vx**2 + vy**2 + vz**2
    dvxdt += (mu[0] * inv_r3 / CC**2)*( (4.0*mu[0]/r - v2) * x + 4.0*r_dot_v * vx )
    dvydt += (mu[0] * inv_r3 / CC**2)*( (4.0*mu[0]/r - v2) * y + 4.0*r_dot_v * vy )
    dvzdt += (mu[0] * inv_r3 / CC**2)*( (4.0*mu[0]/r - v2) * z + 4.0*r_dot_v * vz )
    # <<<
    # >>> non-gravitational component of the acceleration 
    # unit vector in R direction
    ur = [x/r, y/r, z/r]
    H = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx]
    h = hy.sqrt( sum(h_i * h_i for h_i in H) )
    # unit vector in W direction
    uw = [h_i / h for h_i in H]
    # unit vector in S direction
    us = [uw[1] * ur[2] - uw[2] * ur[1],
          uw[2] * ur[0] - uw[0] * ur[2],
          uw[0] * ur[1] - uw[1] * ur[0]]
    # NG coefficients
    alpha, r0, nm, nn, nk = ng_model
    # symmetric radial dependence
    g = alpha * (r/r0)**(-nm) * ( 1.0 + (r/r0)**nn )**(-nk) * cs.NGASCALE
    # implement in equations of motion
    if ng_npars == 0:
        # no NG acceleration
        all_vars = [x, y, z, vx, vy, vz]
    elif ng_npars == 1:
        # only radial component of NGA
        dvxdt += g * (hy.par[0] * ur[0])
        dvydt += g * (hy.par[0] * ur[1])
        dvzdt += g * (hy.par[0] * ur[2]) 
        all_vars = [x, y, z, vx, vy, vz, hy.par[0]]
    elif ng_npars == 2:
        # radial and tranverse components of NGA 
        dvxdt += g * (hy.par[0] * ur[0] + hy.par[1] * us[0])
        dvydt += g * (hy.par[0] * ur[1] + hy.par[1] * us[1])
        dvzdt += g * (hy.par[0] * ur[2] + hy.par[1] * us[2])
        all_vars = [x, y, z, vx, vy, vz, hy.par[0], hy.par[1]]
    elif ng_npars == 3:
        # radial, transverse, and normal, symmetric model
        dvxdt += g * (hy.par[0] * ur[0] + hy.par[1] * us[0] + hy.par[2] * uw[0])
        dvydt += g * (hy.par[0] * ur[1] + hy.par[1] * us[1] + hy.par[2] * uw[1])
        dvzdt += g * (hy.par[0] * ur[2] + hy.par[1] * us[2] + hy.par[2] * uw[2])         
        all_vars = [x, y, z, vx, vy, vz, hy.par[0], hy.par[1], hy.par[2]] 
    elif ng_npars == 4:
        # radial, transverse, and normal, non-symmetric model
        DT = -hy.par[3]
        u = mu[0] / r**3
        p = (x * vx + y * vy + z * vz) / r**2
        q = (vx**2 + vy**2 + vz**2) / r**2 - u
        F0 = 1.
        F1 = 0.
        F2 =-0.5 * u
        F3 = 0.5 * u * p
        F4 = (3. * u * q - 15. * u * p**2 + u**2)/24.
        F5 = (7. * u * p**3 - 3. * u * p * q  - u**2 * p)/8.
        F6 = (630. * u * p**2 * q - 24. * u**2 * q - u**3 - 45. * u * q**2 
           - 945. * u * p**4 + 210. * u**2 * p**2)/720.
        F = F0 + DT*(F1 + DT*(F2 + DT*(F3 + DT*(F4 + DT*(F5 + DT*F6)))))
        G0 = 0.
        G1 = 1.
        G2 = 0.
        G3 =-u / 6. 
        G4 = 0.25 * u * p
        G5 = (9. * u * q - 45. * u * p**2 + u**2)/120.
        G6 = (210. * u * p**3 - 90. * u * p * q - 15. * u**2 * p)/360.
        G7 = (3150. * u * p**2 * q - 54. * u**2 * q - 225. * u * q**2 
           - 4725. * u * p**4 + 630. * u**2 * p**2 - u**3)/5040.
        G = G0 + DT*(G1 + DT*(G2 + DT*(G3 + DT*(G4 + DT*(G5 + DT*(G6 + DT*G7))))))
        x1 = F * x + G * vx
        y1 = F * y + G * vy
        z1 = F * z + G * vz
        r1 = hy.sqrt(x1**2 + y1**2 + z1**2)
        g = alpha * (r1/r0)**(-nm) * ( 1.0 + (r1/r0)**nn )**(-nk) * cs.NGASCALE
        dvxdt += g * (hy.par[0] * ur[0] + hy.par[1] * us[0] + hy.par[2] * uw[0])
        dvydt += g * (hy.par[0] * ur[1] + hy.par[1] * us[1] + hy.par[2] * uw[1])
        dvzdt += g * (hy.par[0] * ur[2] + hy.par[1] * us[2] + hy.par[2] * uw[2])
        all_vars = [x, y, z, vx, vy, vz, hy.par[0], hy.par[1], hy.par[2], hy.par[3]]
    else:
        print('')
        print('Requested NG acceleration not implemented in Heyoka propagator.')
        print('')
        return
    # <<<
    # form the system with the equations of motion
    sys = [(x, dxdt), (y, dydt), (z, dzdt), (vx, dvxdt), (vy, dvydt), (vz, dvzdt)]
    # add variational equations (state transition matrix, sensitivity matrix components)
    vsys = hy.var_ode_sys(sys, all_vars, order=1)
    # construct Taylor adaptive propagator
    ta = hy.taylor_adaptive(sys, compact_mode=True)
    ta_var = hy.taylor_adaptive(vsys, compact_mode=True)
    return ta, ta_var



def PropagateWithHeyoka(x, et0, et, Rs, heyoka_params, n_tau_iter=2):
    ta = heyoka_params
    idx_bkw = np.where(et <  et0)[0]
    idx_fwd = np.where(et >= et0)[0]
    init_state = np.r_[x[0:6], np.block([np.eye(6), np.zeros((6, len(x)-6))]).flatten()]
    # integrate backward from et0 
    ta.time = et0
    ta.pars[:] = x[6:]
    ta.state[:] = init_state
    cf_bkw = ta.propagate_until(et[0] , c_output=True)[4]
    # integrate forward from et0 
    ta.time = et0
    ta.pars[:] = x[6:]
    ta.state[:] = init_state
    cf_fwd = ta.propagate_until(et[-1], c_output=True)[4]
    # light travel-time iteration
    tau = np.zeros_like(et)
    for j in range(n_tau_iter):
        t_bkw = et[idx_bkw] - tau[idx_bkw]
        t_fwd = et[idx_fwd] - tau[idx_fwd]
        sol = np.vstack((cf_bkw(t_bkw), cf_fwd(t_fwd)))
        Delta = np.linalg.norm(sol[:,:3]-Rs, axis=1)
        tau = Delta / CC
    y = np.reshape(sol[:, :6], (len(et), 6))
    PS = np.reshape(sol[:, 6:], (len(et), 6, len(x)))
    P = PS[:, :, 0:6]
    S = PS[:, :, 6:len(x)]
    return y, P, S
