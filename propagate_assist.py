import numpy as np
import rebound
import assist
import constants as cs


def PropagateWithAssist(x, et0, et, Rs, assist_params, n_tau_iter=2):
    m = len(et)
    tau = np.zeros(m)
    factor = cs.AU / cs.CC / cs.DAYS

    for _ in range(n_tau_iter):
        y, P, S = RunAssist(x, et0, et, tau, assist_params)
        tau = np.linalg.norm(y[:, :3] - Rs, axis=1) * factor

    return y, P, S


def RunAssist(x, et0, et, tau, assist_params):
    # --- unpack input ---
    forces, nongrav_coeffs, planets_eph_file, asteroids_eph_file = assist_params

    t0 = et0
    t = et - tau
    m = len(t)

    # --- parameters ---
    nparms = len(x)-6

    # --- ephemeris ---
    ephem = assist.Ephem(planets_eph_file, asteroids_eph_file)

    # --- initial state (convert heliocentric → SSB) ---
    p0h = rebound.Particle(x=x[0], y=x[1], z=x[2], vx=x[3], vy=x[4], vz=x[5])
    p0 = ephem.get_particle('sun', t0) + p0h

    # --- NG parameters (nominal particle) ---
    params_ngforce = np.zeros(3)
    params_ngforce[:nparms] = x[6:]

    # ============================================================
    # TOTAL NUMBER OF VARIATIONAL PARTICLES
    # 6 → state transition matrix
    # nparms → sensitivity matrix
    # ============================================================
    nvar_total = 6 + nparms

    # --- full parameter array ---
    params_all = np.zeros(3 * (1 + nvar_total))

    # nominal particle
    params_all[0:3] = params_ngforce

    # parameter variations (only for sensitivity particles)
    for k in range(nparms):
        idx = 3 * (1 + 6 + k) + k
        params_all[idx] = 1.0

    # ============================================================
    # initialize simulation
    # ============================================================
    sim = rebound.Simulation()
    sim.add(p0)
    sim.t = t0

    extras = assist.Extras(sim, ephem)
    extras.forces = forces
    extras.particle_params = params_all * cs.NGASCALE
    extras.alpha, extras.r0, extras.nm, extras.nn, extras.nk = nongrav_coeffs

    # ============================================================
    # add variational particles
    # ============================================================
    vp = []

    # --- state transition matrix (δx basis) ---
    axes = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    for j, axis in enumerate(axes):
        vpj = sim.add_variation(testparticle=0, order=1)
        pvar = vpj.particles[0]
        # enforce zero initial condition
        pvar.x = pvar.y = pvar.z = 0.0
        pvar.vx = pvar.vy = pvar.vz = 0.0
        setattr(vpj.particles[0], axis, 1.0)
        vp.append(vpj)

    # --- sensitivity matrix (δA basis) ---
    for k in range(nparms):
        vpj = sim.add_variation(testparticle=0, order=1)
        pvar = vpj.particles[0]
        # enforce zero initial condition
        pvar.x = pvar.y = pvar.z = 0.0
        pvar.vx = pvar.vy = pvar.vz = 0.0
        vp.append(vpj)

    # ============================================================
    # allocate outputs
    # ============================================================
    y = np.zeros((m, 6))
    P = np.zeros((m, 6, 6))
    S = np.zeros((m, 6, nparms))

    # ============================================================
    # integration loop
    # ============================================================
    for i, ti in enumerate(t):
        extras.integrate_or_interpolate(ti)

        # --- nominal particle ---
        p = sim.particles[0]
        s = ephem.get_particle('sun', ti)

        y[i, 0] = p.x - s.x
        y[i, 1] = p.y - s.y
        y[i, 2] = p.z - s.z
        y[i, 3] = p.vx - s.vx
        y[i, 4] = p.vy - s.vy
        y[i, 5] = p.vz - s.vz

        # --- state transition matrix ---
        for j in range(6):
            pvar = vp[j].particles[0]
            P[i, 0, j] = pvar.x
            P[i, 1, j] = pvar.y
            P[i, 2, j] = pvar.z
            P[i, 3, j] = pvar.vx
            P[i, 4, j] = pvar.vy
            P[i, 5, j] = pvar.vz

        # --- sensitivity matrix ---
        for k in range(nparms):
            pvar = vp[6 + k].particles[0]
            S[i, 0, k] = pvar.x
            S[i, 1, k] = pvar.y
            S[i, 2, k] = pvar.z
            S[i, 3, k] = pvar.vx
            S[i, 4, k] = pvar.vy
            S[i, 5, k] = pvar.vz

    return y, P, S
