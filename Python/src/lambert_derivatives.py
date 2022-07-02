import numpy as np
from numpy.linalg import norm
from pykep.core import lambert_problem


def lambert_derivatives(r1, v1, r2, v2, tof, mu, debug=False, num_jacob=[0]):

    # variables
    r1n = norm(r1)
    r2n = norm(r2)
    v1n = norm(v1)
    v2n = norm(v2)
    k = np.array([0, 0, 1])
    smu = np.sqrt(mu)

    # true anamaly difference
    cos_dnu = np.dot(r1, r2)/r1n/r2n
    sin_dnu = np.sign(np.dot(np.cross(r1, r2), k)) * np.sqrt(1 - cos_dnu**2)

    energy = v1n**2/2 - mu/r1n
    a = -mu/2/energy
    h = np.cross(r1, v1)
    e = np.sqrt(1 - norm(h)**2/mu/a)
    p = a * (1 - e**2)

    f = 1 - r2n/p * (1 - cos_dnu)
    fdot = np.sqrt(mu/p) * ((1 - cos_dnu)/sin_dnu) * (1/p * (1 - cos_dnu) - 1/r1n - 1/r2n)
    g = r1n * r2n * sin_dnu/ np.sqrt(mu * p)
    gdot = 1 - r1n/p * (1 - cos_dnu)

    # kepler STM
    alpha = 1/a
    sig1 = np.dot(r1, v1) / smu
    sig2 = np.dot(r2, v2) / smu
    chi = alpha * smu * tof + sig2 - sig1

    U1 = - r2n * r1n * fdot / smu
    U2 = r1n * (1 - f)
    U3 = smu * (tof - g)
    U4 = U1*U3 - 1/2 * (U2**2 - alpha*U3**2)
    U5 = (chi**3/6 - U3)/alpha
    Cbar = 1/smu * (3 * U5 - chi * U4 - smu * tof * U2)

    r1 = r1.reshape(3,1)
    r2 = r2.reshape(3,1)
    v1 = v1.reshape(3,1)
    v2 = v2.reshape(3,1)
    dv = v2 - v1
    dr = r2 - r1

    A = r2n/mu * dv @ dv.T + 1/r1n**3 * (r1n * (1 - f) * r2 @ r1.T + Cbar * v2 @ r1.T) + f * np.eye(3)
    B = r1n/mu * (1- f) * (dr @ v1.T - dv @ r1.T) + Cbar/mu * v2 @ v1.T + g * np.eye(3)
    C = -1/r1n**2 * dv @r1.T - 1/r2n**2 * r2 @ dv.T + \
        fdot * (np.eye(3) - 1/r2n**2 * r2 @ r2.T + 1/(mu * r2n) * (r2 @ v2.T - v2 @ r2.T) @ r2 @ dv.T) \
        - mu * Cbar/r2n**3/r1n**3 * r2 @ r1.T
    D = r1n/mu * dv @ dv.T + 1/r2n**3 * (r1n * (1-f) * r2 @ r1.T - Cbar * r2 @ v1.T) + gdot * np.eye(3)

    r1dot = v1
    r2dot = v2
    v1dot = -mu * r1/r1n**3
    v2dot = -mu * r2/r2n**3

    BinvA = np.linalg.solve(B,A)
    CDBinvA = C - D @ BinvA
    DBinv = (np.linalg.solve(B.T, D.T)).T

    # time
    # dv1_dt1 = - (v1dot + BinvA @ r1dot)  # 3 x 1
    # dv1_dt2 = CDBinvA @ r2dot
    # dv2_dt1 = CDBinvA @ r1dot
    # dv2_dt2 = v2dot - DBinv @ r2dot

    dv1_dt1 = (v1dot + BinvA @ r1dot)  # 3 x 1
    dv1_dt2 = -dv1_dt1
    dv2_dt1 = - CDBinvA @ r1dot
    dv2_dt2 = -dv2_dt1

    dv_dt = np.block([[dv1_dt1, dv1_dt2],
                     [dv2_dt1, dv2_dt2]])  # 6 x 2

    # position and velocity
    dv1_dr1 = - BinvA.T   # 3 x 3
    dv1_dr2 = - CDBinvA.T
    dv2_dr1 = CDBinvA
    dv2_dr2 = DBinv

    dv_dr = np.block([[dv1_dr1, dv1_dr2],
                      [dv2_dr1, dv2_dr2]])  # 6 x 6

    if debug:
        # compare with numerical derivatives
        dv_dt_cat = np.hstack([dv1_dt1.flatten(), dv1_dt2.flatten(), dv2_dt1.flatten(), dv2_dt2.flatten()])
        dv_dr_cat = np.hstack([dv1_dr1.flatten(), dv1_dr2.flatten(), dv2_dr1.flatten(), dv2_dr2.flatten()])
        cat_all = np.hstack([dv_dt_cat, dv_dr_cat])
        ID = 1

        print("ID |   value           | numerical             diff ")
        print("----------------------------------------------------------")
        for s in cat_all:
            num = num_jacob[ID-1]
            print("{0:2d} | {1: 12.10e} | {2:12.10e}   ({3:5.4e})".format(ID, s, num,(s - num) / num))
            ID = ID + 1

    return dv_dt, dv_dr


def lambert_numerical_derivatives(r1, r2, tof, mu):

    def lambert_wrap(r1, r2, t1, t2, mu):
        tof = t2 - t1
        lambert = lambert_problem(r1, r2, tof, mu, False, 0)
        v1 = np.array(lambert.get_v1()[0])
        v2 = np.array(lambert.get_v2()[0])
        return v1, v2

    t1 = 0
    t2 = tof

    eps = 1e-5/2
    v1tp, v2tp = lambert_wrap(r1, r2, t1+eps, t2, mu)   # t1 + -> tof -
    v1tm, v2tm = lambert_wrap(r1, r2, t1-eps, t2, mu)
    dv1_dt1 = (v1tp - v1tm)/2/eps
    dv2_dt1 = (v2tp - v2tm)/2/eps

    v1tp, v2tp = lambert_wrap(r1, r2, t1, t2+eps, mu)  # t2 + -> tof +
    v1tm, v2tm = lambert_wrap(r1, r2, t1, t2-eps, mu)
    dv1_dt2 = (v1tp - v1tm)/2/eps
    dv2_dt2 = (v2tp - v2tm)/2/eps

    dv_dt_cat = np.hstack([dv1_dt1.flatten(), dv1_dt2.flatten(),
                           dv2_dt1.flatten(), dv2_dt2.flatten()])

    dv1_dr1 = np.zeros((3, 3))
    dv2_dr1 = np.zeros((3, 3))
    dv1_dr2 = np.zeros((3, 3))
    dv2_dr2 = np.zeros((3, 3))

    for i in range(3):
        epsvec = np.zeros(3)
        epsvec[i] = eps
        v1rp, v2rp = lambert_wrap(r1 + epsvec, r2, t1, t2, mu)  # t1 + -> tof -
        v1rm, v2rm = lambert_wrap(r1 - epsvec, r2, t1, t2, mu)
        dv1_dr1[:, i] = (v1rp - v1rm)/2/eps
        dv2_dr1[:, i] = (v2rp - v2rm)/2/eps

        v1rp, v2rp = lambert_wrap(r1, r2 + epsvec, t1, t2, mu)  # t1 + -> tof -
        v1rm, v2rm = lambert_wrap(r1, r2 - epsvec, t1, t2, mu)
        dv1_dr2[:, i] = (v1rp - v1rm)/2/eps
        dv2_dr2[:, i] = (v2rp - v2rm)/2/eps

    dv_dr_cat = np.hstack([dv1_dr1.flatten(), dv1_dr2.flatten(),
                           dv2_dr1.flatten(), dv2_dr2.flatten()])
    cat_all = np.hstack([dv_dt_cat, dv_dr_cat])

    dv_dt = np.block([[dv1_dt1, dv1_dt2],
                      [dv2_dt1, dv2_dt2]])  # 6 x 2

    dv_dr = np.block([[dv1_dr1, dv1_dr2],
                      [dv2_dr1, dv2_dr2]])  # 6 x 6

    return dv_dt, dv_dr, cat_all




