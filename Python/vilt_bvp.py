import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ************************************************************************
#  File Name: vilt_bvp.py
#  Coder    : Keidai Iiyama 
#
#  [Abstract]
#    Root solve for the value of rC that gives the correct TOF
#
#  [Input]
#    r_L,r_H: distance from the primary body at low,high encounter
#    tof  : time of flight
#    eta_L: angle between an low encounter and the nonleveraging apse 
#           (counter-clockwise) , rad, [-pi,pi]
#    theta: angle between initial encounter -> second encounter [rad],
#           (counter-clockwise), rad [0, 2*pi]
#    N    : approx number of spacecraft orbit revolution  (N:M vilt: planet M
#           revolutions while spacecraft performs N revolutions:
#              Ex: (Earth-Earth) N = 1, M= 3: approx 3 years)
#    L    : Spacecraft Rev number of Maneuver (L = 1,2,3,..., L <= N)
#           L = i means that DV occures during the ith revolution of the
#           spacecraft 
#    D    : Domain; specifies whether a VILT transfer is exterior(+1) or
#                   interior(-1)
#    C    : Chage; direction  of VILT leveraging transfer is exterior or interior
#    S    : Solution; which solution to solve for, the lower rc(1) or a higher
#                     value(2), if it exists
#    debug: If 1, print figures and intermediate outputs
#
#  [Output]
#    r_C     : distance from the primary body at DV point
#    LowArc  : structure of lower arc parameters
#                - v_Enc  [1x3] velcocity at encounter point
#                - nu_Enc [1x1] true anomaly at encounter point
#                - v_RC   [1x3] velcocity at maneuver point 
#                - nu_RC  [1x1] true anomaly at encounter point
#                - a [1x1] semi-major axis
#                - e [1x1] eccentricity
#                - p [1x1] periapsis
#                - tof [1x1] time of flight
#    HighArc : structure of higher arc parameters (same property as LowArc)
#    F       : number of solutions
#
#  [Reference]
#    - Lantulh, D.V, Russel R.P., and Campagnola S. "Automated Inclution 
#      of V-Infinity Leverging Maneuvers in Gravity-Assist Flyby Tour Design"
#      AAS 12-162
# ************************************************************************


class LambertArc:
    def __init__(self):
        self.v_Enc = 0
        self.nu_Enc = 0
        self.v_RC = 0
        self.nu_RC = 0
        self.a = 0
        self.e = 0
        self.p = 0
        self.tof = 0


class CommonParam:
    def __init__(self, D, C, mu, debug):
        self.D = D
        self.C = C
        self.mu = mu
        self.debug = debug

    def get_param(self):
        return self.D, self.C, self.mu, self.debug


def vilt_bvp(r_L, r_H, eta_L, theta, tof, N, L, D, C, S, mu, debug):
    pi = np.pi
    
    cp = CommonParam(D, C, mu, debug)  # register common param

    if debug:
        print(' ')
        print('[VILT LAMBERT]  TOF:', tof, '  N:', N, '  L:', L, '  D:', D, '  C:', C)

    # etaL = true anomaly of low eccentricity encounter [-pi,pi]
    # theta : [0 2pi]
    # nu_L  : [-pi, 2*pi] -> [-pi, pi]
    # nu_H  : [-3*pi, 3*pi] -> [-pi, pi]

    nu_L = eta_L + pi/2*(1-D)

    # convert to [-pi,pi]
    nu_L, dummy = transform_to_pipi(nu_L)

    nu_H = nu_L + C * theta
    nu_H, dummy = transform_to_pipi(nu_H)

    # initialize solution
    r_C = 0
    DV_norm = 0
    F = 0
    LowArc = LambertArc()
    HighArc = LambertArc()

    # Determine K_E of EACH ARC
    # N = K_L + K_H + 1
    if C == 1:  # Low -> High
        K_L = L - 1  # first arc
        K_H = N - 1 - K_L
    elif C == -1:  # High -> Low
        K_H = L - 1  # first arc
        K_L = N -1 - K_H
    else:
        print('C must be -1 or 1')
        return  r_C,LowArc,HighArc,DV_norm,F

    # DETERMINE O_L, O_H FROM GEOMETRY AND TRANSFER TYPE

    # invalid eta_L
    if (eta_L == -pi) or (eta_L == pi):
        print("Invalid eta_L")
        F = 0
        r_C = 0
        return r_C,LowArc,HighArc,DV_norm,F

    O_L,O_H = compute_O(nu_L, nu_H,cp)

    ## BOUND THE DOMAIN OF INTEREST
    if D == 1:  # exterior VILT
        rC_LB = max([r_L, r_H])
        rC_UB = 1000 * rC_LB
    elif D == -1:  # interior VILT
        eps = 1e-5  # to avoid singlar point
        rC_LB_tmp = max([rCsingular(r_H,nu_H,cp), rCsingular(r_L,nu_L,cp)])
        rC_LB = rC_LB_tmp + eps * rC_LB_tmp
        rC_UB = min([r_L, r_H])
        if rC_LB_tmp >= rC_UB:
            # no solution
            return r_C, LowArc, HighArc, DV_norm, F
        i_iter=1
        while rC_LB > rC_UB:  # eps to big -> decay eps
            rC_LB = rC_LB_tmp + np.power(0.5, i_iter) * eps * rC_LB_tmp
            i_iter += 1
            if i_iter > 10:
                return r_C, LowArc, HighArc, DV_norm, F

    else:
        print('D should be either 1 or -1')
        return r_C,LowArc,HighArc,DV_norm,F

    ## Print solution step
    if debug:
        rC_array  = np.linspace(rC_LB,rC_UB,100)
        TOF_array = np.zeros(rC_array.size)
        for ii in range(rC_array.size):
            TOF_array[ii] = calcTOF(rC_array[ii], r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)

        plt.figure()
        plt.title('intersection check')
        plt.xlabel('RC (LU)')
        plt.ylabel('TOF (TU)')
        plt.ylim(0, 30)
        plt.plot(rC_array,TOF_array,'r')
        plt.axhline(y=tof, color='k', linestyle='--')
        plt.show()

    # Determine the number of solutions (F) and adjust bounds as needed for S
    dTOF_LB = calc_dTOF_dRC(rC_LB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)
    tof_LB = calcTOF(rC_LB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)
    tof_UB = calcTOF(rC_UB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)

    if D == 1:
        i_iter=1
        while np.isnan(tof_UB):  # backtrack until tof_UB does not become NaN
            rC_UB  = np.power(0.95, i_iter) * rC_UB
            tof_UB = calcTOF(rC_UB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)

    # time of flight not calculatable at boundary
    if (np.isnan(tof_LB)) or (np.isnan(tof_UB)):
        F = 0
        r_C = 0
        if debug:
            print('tof is nan')
        return r_C,LowArc,HighArc,DV_norm,F

    if dTOF_LB >= 0:
        if (tof_LB > tof) or (tof_UB < tof): # no solution
            F = 0
            r_C = 0
            return r_C,LowArc,HighArc,DV_norm,F
        else:
            F = 1
    else:
        dTOF_UB = calc_dTOF_dRC(rC_UB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)

        if dTOF_UB >= 0:

            if debug:
                print("rC_LB:", rC_LB, "  rc_UB:", rC_UB)
                r_C_min = optimize.fminbound(calcTOF, rC_LB,rC_UB,
                                             args= (r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp), disp=1)
            else:
                r_C_min = optimize.fminbound(calcTOF,rC_LB,rC_UB,
                                             args = (r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp), disp=0)
            tof_min = calcTOF(r_C_min, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)

            if ~np.isreal(tof_min):
                if debug:
                    print('Minumum tof is imaginary')

            if (tof_min > tof) or (tof_UB < tof):  # no solution
                F = 0
                r_C = 0
                return r_C,LowArc,HighArc,DV_norm,F
            elif tof_min == tof:
                F = 1
                r_C = r_C_min
            elif tof_LB < tof:
                F = 1
                rC_LB = r_C_min
            else:
                F = 2
                # adjust the bound to encapture the desired solution
                if S == 1:
                    rC_UB = r_C_min
                elif S == 2:
                    rC_LB = r_C_min

        else:  # dTOF_UB < 0 -> Not robust regions
            F = 0
            r_C = 0
            return r_C,LowArc,HighArc,DV_norm,F

    if debug:
        print('[Bounds]   NSOL: ', F,'  RC_LB = ', rC_LB,'  RC_UB = ', rC_UB)

    # SOLVE FOR RC
    is_solved = 0
    method = 'fzero'
    G_UB = calc_G(rC_UB,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H,cp)
    G_LB = calc_G(rC_LB,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H,cp)
    if (G_LB * G_UB) > 0:
        if debug:
            print('G(UB) and G(LB) has the same value')

    if method == 'Newton-Raphson':
        rC_init = (rC_LB + rC_UB)/2    # initial guess for rC
        r_C, is_solved = solveRC(rC_init,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H,cp)

    # solve using fzero
    elif (method == 'fzero') or (~is_solved):
        def optFunc2(r_C):
            return calc_G(r_C,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H,cp)

        if debug:
            r_C = optimize.bisect(optFunc2,rC_LB,rC_UB,disp=True)
        else:
            r_C = optimize.bisect(optFunc2,rC_LB,rC_UB,disp=False)

    if debug:
       print('rC = ',r_C)
       print(' ')

    ## CALCULATE IN AND OUT VELOCITY AND FLIGHT PATH ANGLE
    v_L, a_L, e_L, p_L = calcVelocity(r_C,r_L,nu_L,cp)
    v_H, a_H, e_H, p_H = calcVelocity(r_C,r_H,nu_H,cp)

    # invalid situation
    # When this happens, trajectory seems to diverge (does not maintain a
    # ellipse orbit)
    if e_L > e_H:
        # disp('  Eccentricity Upside down')
        F = 0
        r_C = 0
        return r_C,LowArc,HighArc,DV_norm,F

    ## Result Structure
    LowArc.v_Enc     = v_L
    LowArc.nu_Enc    = nu_L
    LowArc.v_RC      = np.array([0, -D * np.sqrt(mu/p_L) * (1 + e_L * np.cos(pi/2 * (1+D))), 0])
    LowArc.nu_RC     = pi/2 * (D + 1)
    LowArc.a         = a_L
    LowArc.e         = e_L
    LowArc.p         = p_L
    LowArc.tof       = calcT(r_C, r_L, nu_L, O_L, K_L, cp)

    HighArc.v_Enc    = v_H 
    HighArc.nu_Enc   = nu_H
    HighArc.v_RC     = np.array([0, -D * np.sqrt(mu/p_H) * (1 + e_H * np.cos(pi/2 * (1+D))), 0])
    HighArc.nu_RC    = pi/2 * (D + 1)
    HighArc.a        = a_H
    HighArc.e        = e_H
    HighArc.p        = p_H
    HighArc.tof      = calcT(r_C, r_H, nu_H, O_H, K_H, cp)

    DV_norm = (np.linalg.norm(LowArc.v_RC) - np.linalg.norm(HighArc.v_RC))

    ###### finished function ############
    return r_C,LowArc,HighArc,DV_norm,F

# ****************************************
## SUPPLEMENTARY FUNCTIONS
# ****************************************
def transform_to_pipi(input_angle):
    pi = np.pi
    revolutions = int((input_angle + np.sign(input_angle) * pi) / (2 * pi))

    p1 = truncated_remainder(input_angle + np.sign(input_angle) * pi, 2 * pi)
    p2 = (np.sign(np.sign(input_angle)
                  + 2 * (np.sign(np.fabs((truncated_remainder(input_angle + pi, 2 * pi))
                                      / (2 * pi))) - 1))) * pi

    output_angle = p1 - p2

    return output_angle, revolutions

def truncated_remainder(dividend, divisor):
    divided_number = dividend / divisor
    divided_number = \
        -int(-divided_number) if divided_number < 0 else int(divided_number)

    remainder = dividend - divisor * divided_number

    return remainder

# return O_L,O_H based on C,D,nu_L,nu_H
def compute_O(nu_L,nu_H, cp):
    # nu_case
    #   case 1: nu_L < 0 nu_H < 0
    #   case 2: nu_L < 0 nu_H > 0
    #   case 3: nu_L > 0 nu_H < 0
    #   case 4: nu_L > 0 nu_H > 0
    nu_case = compute_nu_case(nu_L,nu_H)
    D, C, mu, debug = cp.get_param()

    if D == +1:
       if C == +1:
            if nu_case == 1:
                  O_L = +1
                  O_H = -1
            elif nu_case == 2:
                  O_L = +1
                  O_H = +1            
            elif nu_case == 3:
                  O_L = -1
                  O_H = -1              
            elif nu_case == 4:
                  O_L = -1
                  O_H = +1              
       elif C == -1:
            if nu_case == 1:
                  O_L = -1
                  O_H = +1
            elif nu_case == 2:
                  O_L = -1
                  O_H = -1             
            elif nu_case == 3:
                  O_L = +1
                  O_H = +1             
            elif nu_case == 4:
                  O_L = +1
                  O_H = -1            
    elif D == -1:
        if C == +1:
            if nu_case == 1:
                  O_L = -1
                  O_H = +1
            elif nu_case == 2:
                  O_L = -1
                  O_H = -1                
            elif nu_case == 3:
                  O_L = +1
                  O_H = +1                
            elif nu_case == 4:
                  O_L = +1
                  O_H = -1         
        elif C == -1:
            if nu_case == 1:
                O_L = +1
                O_H = -1
            elif nu_case == 2:
                O_L = +1
                O_H = +1                
            elif nu_case == 3:
                O_L = -1
                O_H = -1                 
            elif nu_case == 4:
                O_L = -1
                O_H = +1            

    return O_L, O_H

# compute nu pattern
def compute_nu_case(nu_L,nu_H):
    if (nu_L <= 0) and (nu_H <= 0):
        nu_case = 1
    elif (nu_L <= 0) and (nu_H >= 0):
        nu_case = 2
    elif (nu_L >= 0) and (nu_H <= 0):
        nu_case = 3
    elif (nu_L >= 0) and (nu_H >= 0):
        nu_case = 4

    return nu_case

# Calculate radius of singular point
def rCsingular(r_E, nu_E, cp):
    D, C, mu, debug = cp.get_param()
    r_C = r_E/2 * (1 - D * np.cos(nu_E))
    return r_C

# Calculate time of flight for single arc
def calcT(r_C, r_E, nu_E, O_E, K_E, cp):
    D, C, mu, debug = cp.get_param()

    abs_nu_E = abs(nu_E)
#     if abs_nu_E > pi
#         abs_nu_E = 2 * pi - abs_nu_E;
#     end
    
    e_E =  (r_C - r_E) / (r_E*np.cos(abs_nu_E) + D * r_C)
             
    if (e_E <0) or (e_E > 1):
        t_E = np.nan
        return t_E
             
    a_E = r_C * (r_E * np.cos(abs_nu_E) + D * r_C) / (r_E * np.cos(abs_nu_E) + D*(2*r_C - r_E))
            
    E_E = 2 * np.arctan(np.tan(abs_nu_E/2) * np.sqrt((1-e_E)/(1+e_E)) ) # [-pi/2, pi/2]
    n_E = np.sqrt(mu/a_E**3)
    T_E = 2*np.pi/n_E
    t_PE = (E_E - e_E * np.sin(E_E)) / n_E
    t_E = t_PE * D * O_E + T_E * (K_E + 1/2 - O_E/4 * (D - 1))

    return t_E

# Calculate derivative dTOF/dRC
def calc_dTOF_dRC(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H,cp):
    dTL_dRC = calc_dTE_dRC(r_C,r_L,nu_L,O_L,K_L,cp)
    dTH_dRC = calc_dTE_dRC(r_C,r_H,nu_H,O_H,K_H,cp)
    dTOF_dRC = dTL_dRC + dTH_dRC
    return dTOF_dRC

# Calculate derivative dTE/dRC
# Generated equation using symbolic math toolbox
# -pi < nu_E < pi
def calc_dTE_dRC(r_C,r_E,nu_E,O_E,K_E, cp):
    D, C, mu, debug = cp.get_param()
    nu_E = abs(nu_E)
    pi = np.pi
  
    dTE_dRC = (pi*((3*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**4*(D*r_C + r_E*np.cos(nu_E))**3) \
        - (6*D*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**2)/(r_C**3*(D*r_C + r_E*np.cos(nu_E))**3) \
        + (3*D*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**3*(D*r_C + r_E*np.cos(nu_E))**4)) \
        *(K_E - (O_E*(D - 1))/4 + 1/2))/((mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3) \
        /(r_C**3*(r_C*D + r_E*np.cos(nu_E))**3))**(3/2) - D*O_E*(np.sin(2*np.arctan(np.tan(nu_E/2) \
        *(-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1)/((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2))) \
        /((D*r_C + r_E*np.cos(nu_E))*((mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**3*(r_C*D + r_E*np.cos(nu_E))**3))**(1/2)) \
        - (np.tan(nu_E/2)*((1/(D*r_C + r_E*np.cos(nu_E)) - (D*(r_C - r_E))/(D*r_C + r_E*np.cos(nu_E))**2)/((r_C - r_E) \
        /(D*r_C + r_E*np.cos(nu_E)) + 1) - ((1/(D*r_C + r_E*np.cos(nu_E)) \
        - (D*(r_C - r_E))/(D*r_C + r_E*np.cos(nu_E))**2)*((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1)) \
        /((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1)**2))/((-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1) \
        /((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2)*((np.tan(nu_E/2)**2*((r_C - r_E) \
        /(D*r_C + r_E*np.cos(nu_E)) - 1))/((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1) - 1)) \
        - (D*np.sin(2*np.arctan(np.tan(nu_E/2)*(-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1) \
        /((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2)))*(r_C - r_E)) \
        /((D*r_C + r_E*np.cos(nu_E))**2*((mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3) \
        /(r_C**3*(r_C*D + r_E*np.cos(nu_E))**3))**(1/2)) + (np.sin(2*np.arctan(np.tan(nu_E/2) \
        *(-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1)/((r_C - r_E) \
        /(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2)))*(r_C - r_E) \
        *((3*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**4*(D*r_C + r_E*np.cos(nu_E))**3) \
        - (6*D*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**2)/(r_C**3*(D*r_C + r_E*np.cos(nu_E))**3) \
        + (3*D*mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**3*(D*r_C + r_E*np.cos(nu_E))**4))) \
        /(2*(D*r_C + r_E*np.cos(nu_E))*((mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3) \
        /(r_C**3*(r_C*D + r_E*np.cos(nu_E))**3))**(3/2)) + (np.tan(nu_E/2)*np.cos(2*np.arctan(np.tan(nu_E/2) \
        *(-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1)/((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2))) \
        *((1/(D*r_C + r_E*np.cos(nu_E)) - (D*(r_C - r_E))/(D*r_C + r_E*np.cos(nu_E))**2)/((r_C - r_E) \
        /(D*r_C + r_E*np.cos(nu_E)) + 1) - ((1/(D*r_C + r_E*np.cos(nu_E)) - (D*(r_C - r_E)) \
        /(D*r_C + r_E*np.cos(nu_E))**2)*((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1)) \
        /((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1)**2)*(r_C - r_E)) \
        /((D*r_C + r_E*np.cos(nu_E))*(-((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) - 1) \
        /((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1))**(1/2)*((np.tan(nu_E/2)**2*((r_C - r_E) \
        /(D*r_C + r_E*np.cos(nu_E)) - 1))/((r_C - r_E)/(D*r_C + r_E*np.cos(nu_E)) + 1) - 1) \
        *((mu*(r_E*np.cos(nu_E) + D*(2*r_C - r_E))**3)/(r_C**3*(r_C*D + r_E*np.cos(nu_E))**3))**(1/2)))

    return dTE_dRC

# Caculate total time of flight

def calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp):
    t_L = calcT(r_C, r_L, nu_L, O_L, K_L, cp)
    t_H = calcT(r_C, r_H, nu_H, O_H, K_H, cp)
   
    if (~np.isnan(t_L)) and (~np.isnan(t_H)):
        tof = t_L + t_H
    else:
        tof = np.nan

    return tof

# Solve RC using newton-raphson
def solveRC(rC_init, tof, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp):

    D, C, mu, debug = cp.get_param()
    tol = 1e-10
    maxIterN = 100
    is_solved = 0
    alpha = 0.5 # step coefficient
    
    r_C = rC_init
    
    if debug:
        print('  [Newton-Raphson]')
    
    for i in range(maxIterN):
        G = calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp) - tof
        dG_dRC = calc_dTOF_dRC(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp)
        dRC = G/dG_dRC
        r_C_new = r_C - alpha * dRC
        
        rate = alpha
        while r_C_new < 0:
            rate = rate * alpha
            r_C_new = r_C - rate*dRC
        r_C = r_C_new
        
        if debug:
            print('    i: ',i,' rC = ',r_C,'  |G| = ', G,'  dR_C= ',dRC)
        
        if abs(G) < tol:
            if debug:
              print('  Converged!!')
            is_solved = True
            break

    return r_C, is_solved

# Calculate difference between TOF estimate and calculated TOF
def calc_G(r_C,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H, cp):
   G = calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H, cp) - tof
   return G

# caclulate Velocity in PQW frame

def calcVelocity(r_C,r_E,nu_E, cp):
    D, C, mu, debug = cp.get_param()
    r_hat = np.array([np.cos(nu_E), np.sin(nu_E), 0])
    v_hat = np.array([-np.sin(nu_E),np.cos(nu_E), 0])
    a_E = r_C * (r_E * np.cos(nu_E) + D * r_C) / (r_E * np.cos(nu_E) + D*(2*r_C - r_E))
    e_E = (r_C - r_E)/(r_E*np.cos(nu_E) + D * r_C)
    p_E = a_E * (1 - e_E**2)
    rdot_Norm  =  np.sqrt(mu/p_E) * e_E * np.sin(nu_E)
    rvdot_Norm =  np.sqrt(mu/p_E) * (1 + e_E * np.cos(nu_E))
    v_E = rdot_Norm * r_hat + rvdot_Norm * v_hat

    return v_E, a_E, e_E, p_E


def plot_VILT_PQW(mu, C, D, r_L, r_H, r_C, LowArc, HighArc):
    '''
    Plot the VILT transfer in the PQW frame
    '''
    nu_L = LowArc.nu_Enc
    nu_H = HighArc.nu_Enc
    
    if D == 1:
        domain = 'exterior'
    elif D == -1:
        domain = 'interior'

    if C == 1:  # Low -> high
        y0_arc1   = np.hstack([np.array([r_L*np.cos(nu_L), r_L*np.sin(nu_L), 0]), LowArc.v_Enc])
        yend_arc2 = np.hstack([np.array([r_H*np.cos(nu_H), r_H*np.sin(nu_H), 0]), HighArc.v_Enc])
        # D=+1 apoapsis, D = -1 periapsis
        y0_arc2 = np.hstack([np.array([-D * r_C, 0, 0]), HighArc.v_RC])
        tof_arc1 = LowArc.tof
        tof_arc2 = HighArc.tof
        arcLabel1 = 'Low'
        arcLabel2 = 'High'
    elif C == -1:   # High -> Low
        y0_arc1 = np.hstack([np.array([r_H*np.cos(nu_H), r_H*np.sin(nu_H), 0]), HighArc.v_Enc])
        yend_arc2 = np.hstack([np.array([r_L*np.cos(nu_L), r_L*np.sin(nu_L), 0]),  LowArc.v_Enc])
        # D=+1 apoapsis, D = -1 periapsis
        y0_arc2 = np.hstack([np.array([-D * r_C, 0, 0]), LowArc.v_RC])
        tof_arc1 = HighArc.tof
        tof_arc2 = LowArc.tof
        arcLabel1 = 'High'
        arcLabel2 = 'Low'
    
    # propagate trajectory
    tarc1, yarc1 = twobody(y0_arc1, (0, tof_arc1), mu)
    tarc2, yarc2 = twobody(y0_arc2, (0, tof_arc2), mu)
    
    ## plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # plot center planet
    r = 0.02
    phi = np.deg2rad(range(360))
    plt.plot(r*np.cos(phi), r*np.sin(phi), 'k')

    # plot planets
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(r_L * np.cos(theta), r_L * np.sin(theta), 'k--')
    plt.plot(r_H * np.cos(theta), r_H * np.sin(theta), 'k--')

    
    # plt.title(['VILT Trajectory (', domain, ') ', arcLabel1, ' -> ', arcLabel2,  \
    #     '    nu_L = ', np.rad2deg(nu_L), ' deg', \
    #     '  nu_H = ', np.rad2deg(nu_H), ' deg'])
    plt.plot(yarc1[0,0], yarc1[0,1],'ro',label='Start')
    plt.text(yarc1[0,0] + 0.1, yarc1[0,1],'Start Point')
    plt.plot(yend_arc2[0], yend_arc2[1],'bo', label='End')
    plt.text(yend_arc2[0] + 0.1, yend_arc2[1],'End Point')
    plt.plot(-D * r_C, 0,'go', label='Maneuver')
    plt.text(-D * r_C + 0.1, 0, 'Maneuver Point')
    plt.plot(yarc1[:,0], yarc1[:,1], 'r', label='1st arc')
    plt.plot(yarc2[:,0], yarc2[:,1], 'b', label='2nd arc')
    ax.set_aspect('equal', 'box')
    # plt.legend()
    plt.show()


def twobody(y0, tspan, mu):
    #
    # y             - a matrix whose 12 columns are, respectively,
    #                 X1,Y1,Z1, X2,Y2,Z2, VX1,VY1,VZ1, VX2,VY2,VZ2
    # XG,YG,ZG      - column vectors containing the X,Y and Z coordinates (km)
    #                 the center of mass at the times in t
    # User M-function required:   rkf45
    # User subfunctions required: rates, output
    # ----------------------------------------------------------------------
    teval = np.linspace(tspan[0], tspan[1], 500)
    sol = solve_ivp(rates, tspan, y0, method='DOP853', t_eval=teval, args=(mu,))

    return sol.t, sol.y.T   # N_STATE x T -> T x N_STATE


def rates(t, y, mu):
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    #   This function calculates the accelerations in Equations 2.19
    #   t     - time
    #   y     - column vector containing the position and velocity vectors
    #            of the system at time t
    #   r     - position vector
    #   v     - velocity vector

    #   R     - magnitude of the position vector

    #   dydt  - column vector containing the velocity and acceleration
    #            vectors of the system at time t
    # ------------------------

    r    = y[:3]
    v    = y[3:]
    R    = np.linalg.norm(r) 
    a    = -mu*r/(R**3)
    dydt = np.hstack([v,a])

    return dydt


def calc_theta(sv1, sv2, C):
    '''
    Calculate the parameters required for the VILT solver from the planet position
    '''

    # Calculate 
    if C == -1: # High -> Low
        r_L = norm(sv2[0:3])
        r_H = norm(sv1[0:3])
        sv1_unit = sv1[0:3]/r_H
        sv2_unit = sv2[0:3]/r_L
    elif C == 1: #low -> High
        r_L = norm(sv1[0:3])
        r_H = norm(sv2[0:3])       
        sv1_unit = sv1[0:3]/r_L
        sv2_unit = sv2[0:3]/r_H      
        
    # calculate sv1 -> sv2 [0 2*pi]
    theta = np.arccos(np.dot(sv1_unit, sv2_unit))  # [0, pi] 
    cross_12 = np.cross(sv1_unit, sv2_unit)
    
    # print for debug
    # disp(['calc theta: ',num2str(rad2deg(theta)), '  cross(3): ', num2str(cross_12(3),'#.4f')]);
    
    if cross_12[2] < 0:  # z < 0: clock wise -> change to counter clock wise
        theta = 2*np.pi - theta

    return r_L, r_H, theta