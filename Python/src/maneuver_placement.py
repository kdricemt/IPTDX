import copy
import sys
sys.path.append("../src")
import numpy as np
from pykep.core import lambert_problem, ic2par, epoch, DAY2SEC, AU, propagate_lagrangian, fb_prop
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler
from scipy import optimize
import matplotlib.pyplot as plt
from numpy.linalg import norm


def calc_true_anomaly(evec, e, R, r, v):
    nu = np.arccos(np.dot(evec, R)/e/r)
    if np.dot(R, v) < 0:
        nu = 2 * np.pi - nu
    return nu

def calc_true_anom_from_rv(R0, v0, mu):
    r0 = np.linalg.norm(R0)
    energy = 1 / 2 * np.linalg.norm(v0) ** 2 - mu / r0
    h = np.cross(R0, v0)
    evec = np.cross(v0, h) / mu - R0 / r0
    e = np.linalg.norm(evec)
    nu0 = calc_true_anomaly(evec, e, R0, r0, v0)

    return np.rad2deg(nu0)


def true_to_mean(nu, e):
    E = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), (e + np.cos(nu)))
    if E < 0:
        E = E + 2*np.pi  # convert to [0,2*np.pi]
    M = E - e * np.sin(E)

    # account for multiple rev
    # Todo: Do we need this?
    M = M + 2 * np.pi * (nu // (2 * np.pi))

    return M


def I1(e, nu, M):
    tmp = 1/(1 - e**2) * ((2 * e)/(1 + e * np.cos(nu)) - np.cos(nu) - (3 * e**2 * M * np.sin(nu))/np.power(1 - e**2, 3/2))
    return tmp


def I2(e, nu, M):
    tmp = 1/(1 - e**2) * ((1 + 1/(1 + e * np.cos(nu))) * np.sin(nu) - (3 * e * M * (1 + e * np.cos(nu)))/np.power(1 - e**2, 3/2))
    return tmp

def rtn_to_catesian(r, v):
    h = np.cross(r, v)
    R = r / np.linalg.norm(r)
    N = h / np.linalg.norm(h)
    T = np.cross(N, R)
    rot = np.vstack([R, T, N]).transpose()  # 3 x 3

    return rot


def manuever_placement(R0, Rf, v0, vf, mu, dJ_dV_func, debug=False):
    """
    Compute optimal manuever placement using primer vector theory

    Ref:
        Section II.C in Landau "Efficient Maneuver Placement for Automated Trajectory Design"

    v0, vf: solution of the lambert problem
    """

    # 2. compute partials of J with respect to v0, vf ----------------
    # Todo: implement this for various objectives
    dJ_dV0, dJ_DVf = dJ_dV_func(v0, vf)  # in RTN frame

    # 3. solve for A - F -----------------------------------------------
    # compute nu
    r0 = np.linalg.norm(R0)
    rf = np.linalg.norm(Rf)
    energy = 1/2 * np.linalg.norm(v0) **2 - mu / r0
    a = - mu / (2 * energy)
    h = np.cross(R0, v0)
    evec = np.cross(v0, h)/mu - R0/r0
    e = np.linalg.norm(evec)

    nu0 = calc_true_anomaly(evec, e, R0, r0, v0)
    nuf = calc_true_anomaly(evec, e, Rf, rf, vf)
    M0 = true_to_mean(nu0, e)
    Mf = true_to_mean(nuf, e)

    r0_p = 1/(1 + e * np.cos(nu0))
    p_r0 = 1/r0_p
    rf_p = 1/(1 + e * np.cos(nuf))
    p_rf = 1/rf_p

    a_rt = np.array([[np.cos(nu0), -e * np.sin(nu0), I1(e, nu0, M0), 0],
                     [-(1 + r0_p)*np.sin(nu0), p_r0, I2(e, nu0, M0), r0_p],
                     [np.cos(nuf), -e * np.sin(nuf), I1(e, nuf, Mf), 0],
                     [-(1 + rf_p) * np.sin(nuf), p_rf, I2(e, nuf, Mf), rf_p]]
                    )
    b_rt = np.array([dJ_dV0[0], dJ_dV0[1], -dJ_DVf[0], -dJ_DVf[1]])
    ABCD = np.linalg.solve(a_rt, b_rt)

    a_n = np.array([[r0_p * np.cos(nu0), r0_p * np.sin(nu0)],
                   [rf_p * np.cos(nuf), rf_p * np.sin(nuf)]])
    b_n = np.array([dJ_dV0[2], -dJ_DVf[2]])
    EF = np.linalg.solve(a_n, b_n)

    A, B, C, D = ABCD[0], ABCD[1], ABCD[2], ABCD[3]
    E, F = EF[0], EF[1]

    # 4. Choose an initial guess for nu
    if nuf < nu0:
        nuf = nuf + 2*np.pi
    else:
        nuf = nuf

    nu_guess_list = []
    if (nu0 <= 2*np.pi) and (nuf >= 2*np.pi):  # contains periapsis within [nu0, nuf]
         nu_guess_list.append(2*np.pi)
    elif ((nu0 <= np.pi) and (nuf >= np.pi)) or ((nu0 >= np.pi) and (nuf >= 3 * np.pi)):  # contains apoapsis within [nu0, nuf]
         nu_guess_list.append(np.pi)

    # 5. maximize primer vector (in RTN)
    def primer_vector(nu):
        M = true_to_mean(nu, e)
        p_R = A * np.cos(nu) - B * e * np.sin(nu) + C * I1(e, nu, M)
        p_T = -A * np.sin(nu) + B * (1 + e * np.cos(nu)) + (D - A * np.sin(nu))/(1 + e * np.cos(nu)) + C * I2(e, nu, M)
        p_N = (E * np.cos(nu) + F * np.sin(nu))/(1 + e * np.cos(nu))
        p = np.array([p_R, p_T, p_N])
        return p

    def primer_vector_minus_norm(nu):
        return -np.linalg.norm(primer_vector(nu))  # the objective is to minimize this

    def primer_vector_jac(nu):
        p = primer_vector(nu)
        r_p = 1/(1 + e * np.cos(nu))
        p_R, p_T, p_N = p[0], p[1], p[2]
        dp_dnu_R = p_T * (1 - r_p) - (A * np.sin(nu) - C * np.sin(nu) + D * e * np.cos(nu)) * r_p**2
        dp_dnu_T = (A * e + p_T * e * np.sin(nu) + D * e * np.cos(nu)) * r_p - 2 * p_R
        dp_dnu_N = (-E * np.sin(nu) + e * F + F * np.cos(nu)) * (r_p)**2
        dp_dnu = np.array([dp_dnu_R, dp_dnu_T, dp_dnu_N])
        df_dnu = -1/np.linalg.norm(p) * p * dp_dnu
        return df_dnu

    # 6. maximize primer vector
    ndiv = 5
    for i in range(ndiv):
        lb = nu0 + (nuf - nu0) * i / ndiv
        ub = nu0 + (nuf - nu0) * (i+1) / ndiv
        res = optimize.minimize_scalar(primer_vector_minus_norm,  bounds=(lb,ub), method='bounded')
        nu_guess = res.x
        nu_guess_list.append(nu_guess)

    # compare the results with apoapsis and periapsis results and adopt that if it is better than the obtained
    # optimal
    p_opt_norm = -np.inf
    if len(nu_guess_list) > 0:
        for nu in nu_guess_list:
            p_guess = primer_vector(nu)
            p_guess_norm = np.linalg.norm(p_guess)
            if p_guess_norm > p_opt_norm:  # change optimal if its better
                p_opt = p_guess
                nu_opt = nu
                p_opt_norm = p_guess_norm

    if debug:
        nus = np.linspace(nu0, nuf, 500)
        pvs = np.zeros_like(nus)
        plt.figure()
        for i, nu in enumerate(nus):
            pvs[i] = np.linalg.norm(primer_vector(nu))
        plt.plot(nus, pvs)
        plt.scatter(nu_opt, np.linalg.norm(p_opt), color=['r'])
        plt.title('Primer Vector Norm')
        plt.xlabel('nu')
        plt.ylabel('|p|')
        plt.show()

    # 7. obtain the optimal maneuver position and primer vector -----------------------------
    theta_0m = nu_opt - nu0  # angle from initial to maneuver point
    theta_0f = nuf - nu0  # angle from initial to final

    p_opt_normalized = p_opt / np.linalg.norm(p_opt)
    dV_hat = p_opt_normalized

    # Create a initial guess of tangential velocity at maneuver position
    p = a * (1 - e**2)
    r_m = p/(1 + e*np.cos(nu_opt))
    V_T1_guess = np.sqrt(p*mu)/r_m  # p = r^2 V_T^2/mu -> V_T = sqrt(p*mu) /r

    if debug:
        M_dv = true_to_mean(nu0 + theta_0m, e)
        t_dv = (M_dv - M0) * np.sqrt(a**3/mu)
        print("T_DV:", t_dv)

    return theta_0m, theta_0f, dV_hat, V_T1_guess


def flight_time_and_velocity(R0, Rf, t0, tf, mu, dV_norm, theta_0m, theta_0f, dV_hat, V_T1_guess, debug):
    """
    Match flight time and compute initial/final velocity given the manuever placement

    Ref:
        Section III.D in Landau "Efficient Maneuver Placement for Automated Trajectory Design"
    """

    # 1: Obtain optimal location of maneuver
    dV = dV_norm * dV_hat
    dV_R, dV_T, dV_N = dV[0], dV[1], dV[2]

    r0 = np.linalg.norm(R0)
    rf = np.linalg.norm(Rf)

    def compute_params(V_T1):
        # 2. Solve for i0
        x = np.sqrt((dV_N * np.cos(theta_0m))**2 + (dV_T + V_T1)**2)
        i0 = 2 * np.arctan((x - dV_N * np.cos(theta_0m))/(dV_T + V_T1))
        tmp = np.arccos(dV_N * np.cos(theta_0f) * np.sin(theta_0m)/(x * np.sin(theta_0f)))
        i0_plus = i0 + tmp
        i0_minus = i0 - tmp
        if np.cos(i0_plus) > np.cos(i0_minus):
            i0 = i0_plus
        else:
            i0 = i0_minus

        # 3. compute V_T2
        cos_theta_mf = np.cos(theta_0f) * np.cos(theta_0m) + np.sin(theta_0f) * np.sin(theta_0m) * np.cos(i0)
        sin_theta_mf = np.sqrt(1 - cos_theta_mf**2)  # assume plus
        tmp2 = (V_T1 + dV_T)/(np.sin(theta_0f) * np.cos(theta_0m) * np.cos(i0) - np.cos(theta_0f) * np.sin(theta_0m))
        if tmp2 < 0:
            sin_theta_mf = -sin_theta_mf  # V_T2 has to be plus
        V_T2 = tmp2 * sin_theta_mf

        # 4. compute manuever location r_m
        theta_mf = np.arctan2(sin_theta_mf, cos_theta_mf)
        co_eff = np.array([
            -(V_T1/(r0 * np.sin(theta_0m))+ V_T2/(rf * sin_theta_mf)),
            V_T1/np.tan(theta_0m) + V_T2 / (sin_theta_mf/cos_theta_mf) - dV_R,
            mu/V_T1 * np.tan(theta_0m/2) + mu/V_T2 * np.tan(theta_mf/2)
        ])
        r_ms = np.roots(co_eff)  # solve quadratic equation
        if r_ms[0] > 0:
            r_m = r_ms[0]
        else:
            r_m = r_ms[1]

        # 5. compute t_0m and t_0f
        h1 = r_m * V_T1
        V_R0 = - V_T1/np.sin(theta_0m) + h1/(r0 * np.tan(theta_0m)) + mu/h1 * np.tan(theta_0m/2)
        V_R1 = h1/(r0 * np.sin(theta_0m)) - V_T1/np.tan(theta_0m) - mu/h1 * np.tan(theta_0m/2)  # TODO: what is r_1? (assumed it's r0)

        alpha_1 = 2 * mu /r0 - V_R0**2 - h1**2/r0**2

        if alpha_1 < 0:
            t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf = -1000, -1000, np.zeros(3), np.zeros(3), np.nan, np.nan, np.nan, np.nan
            return t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf

        # TODO: putted /2 inside parenthis assuming its typo in paper
        chi1 = 2/np.sqrt(alpha_1) * np.arctan((np.sqrt(alpha_1) * r0 * np.tan(theta_0m/2))/(h1 - V_R0 * r0 * np.tan(theta_0m/2)))

        if alpha_1 >0 and chi1>0:
            N1 = np.floor(theta_0m/(2 * np.pi))
        elif alpha_1>0 and chi1 <0:
            N1 = np.floor(theta_0m/(2*np.pi)) + 1
        else:
            N1 = 0

        t_0m = (mu * chi1 + V_R0 * r0 - V_R1*r_m)/alpha_1 + N1*mu/np.sqrt(alpha_1**3)  # time unitl maneuver

        h2 = r_m * V_T2  # TODO: Not mentioned in paper, this is my guess
        V_Rf = V_T2/np.sin(theta_mf) - h2/(rf * np.tan(theta_mf)) - mu/h2 * np.tan(theta_mf/2)
        alpha_2 = 2*mu/rf - V_Rf**2 - h2**2/rf**2   # TODO: what is hf? -> assumed it is h2

        if alpha_2 < 0:
            t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf = -1000, -1000, np.zeros(3), np.zeros(3), 0, 0, 0, 0
            return t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf

        chi2 = 2/np.sqrt(alpha_2) * np.arctan((np.sqrt(alpha_2) * rf * np.tan(theta_mf/2))/(h2 + V_Rf * rf * np.tan(theta_mf/2)))

        if alpha_2 >0 and chi2 > 0:
            N2 = np.floor((theta_0f - theta_0m)/(2 * np.pi))
        elif alpha_2>0 and chi2 < 0:
            N2 = np.floor((theta_0f - theta_0m)/(2*np.pi)) + 1
        else:
            N2 = 0

        t_mf = (mu * chi2 + (V_R1 + dV_R)*r_m - V_Rf*rf)/alpha_2 + N2 * mu/np.sqrt(alpha_2**3)

        if t_0m < 0:
            t_0m = -1000.0
        if t_mf < 0:
            t_mf = -1000.0

        return t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf

    def compute_tof_diff(V_T1):
        """
        Compute the difference between given and calculated tof
        """
        t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf = compute_params(V_T1)
        calc_tof = t_0m + t_mf
        given_tof = tf - t0

        return calc_tof - given_tof

    def compute_tof_diff_with_derivatives(V_T1):
        eps = V_T1 * 1e-6
        diff = compute_tof_diff(V_T1)
        diff_eps = compute_tof_diff(V_T1 + eps)
        diff_dv = (diff_eps - diff)/eps

        return diff, diff_dv

    def compute_eta_and_V(V_T1, succ):

        t_0m, t_mf, V_R0, V_Rf, i0, h1, h2, theta_mf = compute_params(V_T1)
        eta = t_0m/(t_0m + t_mf)
        V0 = np.array([V_R0, np.cos(i0) * h1 / r0, np.sin(i0) * h1/r0])

        tmp = np.sin(i0) * np.sin(theta_0m)/np.sin(theta_mf)
        Vf = np.array([V_Rf, np.sqrt(1 - tmp**2) * h2/rf, -tmp*h2/rf])

        if t_0m < 0 or t_mf < 0:
            succ = False
            if debug:
                print("  Calculated transfer time is minus... t_0m: {0:.5f}   t_mf: {1:.5f}".format(t_0m,t_mf))
        else:
            if debug:
                print("  t_0m: {0:.5f}  t_mf: {1:.5f}  i0 [deg]:{2:.5f}".format(t_0m, t_mf, np.rad2deg(i0)))

        T = t_0m + t_mf

        return succ, eta, V0, Vf, T

    # root solve for V_T1 to match tof'
    try:
        use_newton = False
        skip_for_dv0 = False

        if dV_norm == 0 and skip_for_dv0:
            V_T1_opt = V_T1_guess
            succ = True
        else:
            if use_newton:
                sol = optimize.root_scalar(compute_tof_diff_with_derivatives, x0=V_T1_guess, fprime=True, method='newton')
                # sol = optimize.root_scalar(compute_tof_diff_with_derivatives, x0=0.1 * V_T1_guess, fprime=True, method='newton')
            else:
                lb = 1.0 * V_T1_guess
                tof_diff = compute_tof_diff(lb)
                tof_diff_prev = copy.deepcopy(tof_diff)
                change_lb = True
                while change_lb:
                    if np.isnan(tof_diff) or tof_diff < -500:
                        lb = lb + 0.1 * V_T1_guess
                        ub = 2.0 * V_T1_guess
                        change_lb = False
                    else:
                        if np.sign(tof_diff_prev) != np.sign(tof_diff):
                            ub = lb + 0.1 * V_T1_guess
                            change_lb = False
                        else:  # go to next step
                            tof_diff_prev = copy.deepcopy(tof_diff)
                            lb = lb - 0.1 * V_T1_guess
                            tof_diff = compute_tof_diff(lb)

                    if lb == 0.5*V_T1_guess:
                        ub = 2.0 * V_T1_guess
                        break

                sol = optimize.root_scalar(compute_tof_diff, bracket=(lb, ub), method='brentq')

            V_T1_opt = sol.root
            succ = sol.converged

        # 7. compute velocity in cartesian
        succ, eta, V0, Vf, T = compute_eta_and_V(V_T1_opt, succ)

    except:
        succ = False
        eta = 0.0
        V0 = np.zeros(3)
        Vf = np.zeros(3)
        T = 0.0

    if debug:
        V_T1s = np.linspace(0.1 * V_T1_guess, 2 * V_T1_guess, 500)
        diffs = np.zeros_like(V_T1s)
        plt.figure()
        for i, V_T1 in enumerate(V_T1s):
            diffs[i] = compute_tof_diff(V_T1)
        plt.plot(V_T1s, diffs)
        plt.scatter(V_T1_guess, compute_tof_diff(V_T1_guess), color=['g'], label="initial")
        if succ:
            plt.scatter(lb, compute_tof_diff(lb), color=['b'], label="lb")
            plt.scatter(ub, compute_tof_diff(ub), color=['m'], label="ub")
            plt.scatter(V_T1_opt, compute_tof_diff(V_T1_opt), color=['r'], label="opt")
            print("TOF Diff: ", compute_tof_diff(V_T1_opt))
        plt.title('TOF Diffs')
        plt.xlabel('V_T1')
        plt.ylabel('TOF (calc - given)')
        plt.legend()
        plt.show()

    return succ, eta, V0, Vf, T


def solve_arc(body_0, body_f, T_START, tf, dV_norm_list, mu, objective_type=1, debug=False):
    """
    Generate trajectory arc dataset by varying the DV magnitude
    """
    # scale r, t, mu
    non_dimension = True
    t0 = 0

    if non_dimension:
        r_ref = AU
        t_ref = 2 * np.pi * np.sqrt(AU**3/mu)
    else:
        r_ref = 1
        t_ref = 1

    v_ref = r_ref / t_ref
    mu_ref = r_ref**3/t_ref**2

    mu = mu / mu_ref

    n_dv = dV_norm_list.size
    V0_store = np.zeros((n_dv, 3))
    Vf_store = np.zeros((n_dv, 3))
    DV_store = np.zeros((n_dv, 1))
    DV_vec_store = np.zeros((n_dv, 3))
    calc_tof_store = np.zeros(n_dv)
    eta_store = np.zeros(n_dv)
    n_sol = 0

    # poistion and velocities of the bodies
    t_P0 = epoch(T_START.mjd2000 + t0)
    r_P0, V_P0 = body_0.eph(t_P0)
    r_P0 = np.array(r_P0)/r_ref
    V_P0 = np.array(V_P0)/v_ref
    rmin0 = body_0.radius * 1.1 / r_ref
    mu_0 = body_0.mu_self / mu_ref

    t_Pf = epoch(T_START.mjd2000 + tf)
    r_Pf, V_Pf = body_f.eph(t_Pf)
    r_Pf = np.array(r_Pf) / r_ref
    V_Pf = np.array(V_Pf)/ v_ref
    vpf = np.linalg.norm(V_Pf)
    V_Pf_hat = V_Pf / vpf
    rminf = body_f.radius * 1.1 / r_ref
    mu_f = body_f.mu_self / mu_ref

    # 1. solve lambert problem -----------------------------------------
    tof = (tf - t0)*DAY2SEC/t_ref
    lambert = lambert_problem(r_P0, r_Pf, tof, mu, False, 0)
    if debug:
        plot_lambert(lambert)
        
    v0_l = np.array(lambert.get_v1()[0])   # scaled
    vf_l = np.array(lambert.get_v2()[0])

    rtn_to_cart_0 = rtn_to_catesian(r_P0, v0_l)
    rtn_to_cart_f = rtn_to_catesian(r_Pf, vf_l)

    vp0 = np.linalg.norm(V_P0)
    V_P0_hat = V_P0/vp0

    vpf = np.linalg.norm(V_Pf)
    V_Pf_hat = V_Pf/vpf

    # objective function and its derivatives
    # the inputs are in cartesian!
    def J_Vinf(V0, Vf):
        return np.linalg.norm(Vf - V_Pf) - np.linalg.norm(V0 - V_P0)

    def dJ_dV_minVinf(V0, Vf):
        dJ_dV0 = -(V0 - V_P0)/np.linalg.norm(V0 - V_P0)
        dJ_dVf = (Vf - V_Pf)/np.linalg.norm(Vf - V_Pf)

        return rtn_to_cart_0.transpose() @ dJ_dV0, rtn_to_cart_f.transpose() @ dJ_dVf

    def P0_minus(V0):
        Vinf_0 = V0 - V_P0
        vinf0 = np.linalg.norm(Vinf_0)
        Vinf_0_hat = Vinf_0 / vinf0
        alpha0_plus = np.arccos(np.dot(Vinf_0_hat, V_P0_hat))
        delta_0 = 2 * np.arcsin(1/(1 + rmin0 * vinf0**2/mu_0))
        alpha0_minus = alpha0_plus - delta_0
        V0_2 = vp0**2 + 2 * vp0 * vinf0 * np.cos(alpha0_minus) + vinf0**2
        res = np.pi * mu/np.sqrt(2) * np.power(mu/rmin0 - V0_2/2, -3/2)
        return res

    def Pf_plus(Vf):
        Vinf_f = Vf - V_Pf
        vinff = np.linalg.norm(Vinf_f)
        Vinf_f_hat = Vinf_f / vinff
        alphaf_minus = np.arccos(np.dot(Vinf_f_hat, V_Pf_hat))
        delta_f = 2 * np.arcsin(1 / (1 + rminf * vinff ** 2 / mu_f))
        alphaf_plus = alphaf_minus + delta_f
        Vf_2 = vpf ** 2 + 2 * vpf * vinff * np.cos(alphaf_plus) + vinff ** 2
        res = np.pi * mu / np.sqrt(2) * np.power(mu / rminf - Vf_2 / 2, -3 / 2)
        return res

    def J_minTOF(V0, Vf):
        J = Pf_plus(Vf) - P0_minus(V0)
        return J

    def dJ_dV_minTOF(V0, Vf):
        P0_minus_original = P0_minus(V0)
        Pf_plus_original = Pf_plus(Vf)
        eps = 1e-6

        dJ_dV0 = np.zeros(3)
        dJ_dVf = np.zeros(3)
        for i in range(3):
            V0_tmp = copy.deepcopy(V0)
            V0_tmp[i] += eps
            P0_minus_perturb = P0_minus(V0_tmp)
            dJ_dV0[i] = - (P0_minus_perturb - P0_minus_original)/eps
        for i in range(3):
            Vf_tmp = copy.deepcopy(Vf)
            Vf_tmp[i] += eps
            Pf_plus_perturb = Pf_plus(Vf_tmp)
            dJ_dVf[i] = (Pf_plus_perturb - Pf_plus_original)/eps

        return rtn_to_cart_0.transpose() @ dJ_dV0, rtn_to_cart_0.transpose() @ dJ_dVf

    # first solve for maneuver placement
    if objective_type==1:
        # Minimum Vinf
        J_func = J_Vinf
        dJ_dV_func = dJ_dV_minVinf
    elif objective_type==2:
        # Minimum TOF trajectory
        J_func = J_minTOF
        dJ_dV_func = dJ_dV_minTOF
    else:
        dJ_dV_func = None

    # solve for maneuver placement
    theta_0m, theta_0f, dV_hat, V_T1_guess = manuever_placement(r_P0, r_Pf, v0_l, vf_l, mu, dJ_dV_func, debug)

    if debug:
        print("Primer Vector Direction (RTN): ", dV_hat)
        print("theta_0m: {0:.2f}  theta_0f: {1:.2f}".format(np.rad2deg(theta_0m), np.rad2deg(theta_0f)))

    v0_l = v0_l * v_ref  # with units
    vf_l = vf_l * v_ref

    # iterate for different delta V norms
    debug_tmp = copy.deepcopy(debug)
    for dV_norm in dV_norm_list:
        if debug:
            print("DV: {0:.2f}".format(dV_norm))
        succ, eta, V0, Vf, calc_tof = flight_time_and_velocity(r_P0, r_Pf, t0*DAY2SEC/t_ref, tf*DAY2SEC/t_ref,
                                                               mu, dV_norm/v_ref, theta_0m, theta_0f, dV_hat, V_T1_guess, debug_tmp)
        if succ:
            calc_tof = calc_tof * t_ref
            eta_store[n_sol] = eta
            V0 = V0 * v_ref
            Vf = Vf * v_ref
            V0_cart = rtn_to_catesian(r_P0 * r_ref, v0_l) @ V0
            Vf_cart = rtn_to_catesian(r_Pf * r_ref, vf_l) @ Vf
            V0_store[n_sol,:] = V0_cart
            Vf_store[n_sol,:] = Vf_cart
            DV_store[n_sol] = dV_norm
            calc_tof_store[n_sol] = calc_tof
            DV_vec_store[n_sol,:] = dV_norm * dV_hat
            n_sol += 1
            # debug_tmp = False   # only debug once

            if debug:
                # compute objective
                J_lambert = J_func(v0_l / v_ref, vf_l / v_ref)
                J_dv = J_func(V0_cart / v_ref, Vf_cart / v_ref)
                print("J  lambert{0:.3e}  DV:{1:.3e}".format(J_lambert, J_dv))

                # compare ballistic trajectory (lambert) and DV trajectory
                nu0_dv = calc_true_anom_from_rv(r_P0 * r_ref, V0_cart, mu=mu * mu_ref)
                nuf_dv = calc_true_anom_from_rv(r_Pf * r_ref, Vf_cart, mu=mu * mu_ref)
                nu0_l = calc_true_anom_from_rv(r_P0 * r_ref, v0_l, mu=mu * mu_ref)
                nuf_l = calc_true_anom_from_rv(r_Pf * r_ref, vf_l, mu=mu * mu_ref)

                tof = (tf - t0) * DAY2SEC
                r_dv, v_dv = propagate_lagrangian(r_P0 * r_ref, V0_cart, eta*tof, mu=mu * mu_ref)
                nu_dv_dv = calc_true_anom_from_rv(r_dv, v_dv, mu=mu * mu_ref)

                nu_dv_l = nu0_l + np.rad2deg(theta_0m)

                print("  eta: {0:.3f}".format(eta))
                print("  V0 Norm  Lambert: {0:.3f}  V: {1:.3f}".format(norm(v0_l), norm(V0)))
                print("  V Norm  Lambert: {0:.3f}  V: {1:.3f}".format(norm(vf_l), norm(Vf)))
                print("  Initial  nu (Lambert):", nu0_l)
                print("  Initial  nu (DV)     :", nu0_dv)
                print("  DV Point nu (Lambert):", nu_dv_l)
                print("  DV Point nu (DV)     :", nu_dv_dv)
                print("  Final    nu (Lambert):", nuf_l)
                print("  Final    nu (DV)     :", nuf_dv)

                if dV_norm == 0.0:
                    print(" is zero")


    if n_sol > 0:
        succ = True
        eta_store = eta_store[:n_sol]
        V0_store = V0_store[:n_sol,:]
        Vf_store = Vf_store[:n_sol, :]
        DV_store = DV_store[:n_sol]
    else:
        succ = False
        eta_store = np.nan
        V0_store = np.nan * np.ones(3)
        Vf_store = np.nan * np.ones(3)
        DV_store = np.nan

    Vinf_0_store = V0_store - V_P0 * v_ref
    Vinf_f_store = Vf_store - V_Pf * v_ref

    if debug and n_sol > 0:
        # Create the figure and axis
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter([0], [0], [0], color=['y'])

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter([0], [0], [0], color=['y'])
        ax2.view_init(90, 0)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.scatter([0], [0], [0], color=['y'])
        ax3.view_init(0, 0)

        for ax in [ax1, ax2, ax3]:
            # Plot the planet orbits
            plot_planet(body_0, t0=t_P0, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)
            plot_planet(body_f, t0=t_Pf, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)
            # Plot the Lambert solutions

            # propagate until DV
            plot_idx = 0
            tof = (tf-t0)*DAY2SEC

            r_dv, v_dv = propagate_lagrangian(r_P0 * r_ref, V0_store[plot_idx,:], eta_store[plot_idx] * tof, mu=mu * mu_ref)
            plot_kepler(r_P0 * r_ref, V0_store[plot_idx,:], eta_store[plot_idx]*tof, color='r', N=500, units=AU, mu=mu*mu_ref, axes=ax, label='arc1')

            v_dv_add = rtn_to_catesian(r_dv, v_dv) @ DV_vec_store[plot_idx,:]
            v_dv += v_dv_add + v_dv
            # plot_kepler(r_dv, v_dv, (1 - eta_store[0]) * tof, color='g', N=500, units=AU, mu=mu * mu_ref, axes=ax, label='arc2')
            plot_kepler(r_Pf * r_ref, -Vf_store[plot_idx,:], (1-eta_store[plot_idx])*tof, color='g', N=500, units=AU, mu=mu*mu_ref, axes=ax, label='arc2')


        plt.show()


    return succ, eta_store, V0_store, Vf_store, DV_store, Vinf_0_store, Vinf_f_store

















