% ************************************************************************
%  File Name: vilt_bvp
%  Coder    : Keidai Iiyama 
%
%  [Abstract]
%    Root solve for the value of rC that gives the correct TOF
%
%  [Input]
%    r_L,r_H: distance from the primary body at low,high encounter
%    tof  : time of flight
%    eta_L: angle between an low encounter and the nonleveraging apse 
%           (counter-clockwise) , rad, [-pi,pi]
%    theta: angle between initial encounter -> second encounter [rad],
%           (counter-clockwise), rad [0, 2*pi]
%    N    : approx number of spacecraft orbit revolution  (N:M vilt: planet M
%           revolutions while spacecraft performs N revolutions:
%              Ex: (Earth-Earth) N = 1, M= 3: approx 3 years)
%    L    : Spacecraft Rev number of Maneuver (L = 1,2,3,..., L <= N)
%           L = i means that DV occures during the ith revolution of the
%           spacecraft 
%    D    : Domain; specifies whether a v‡ transfer is exterior(+1) or
%                   interior(-1)
%    C    : Chage; direction  of V‡ leveraging transfer is exterior or interior
%    S    : Solution; which solution to solve for, the lower rc(1) or a higher
%                     value(2), if it exists
%    units: non-dimenionalization units (L,V,T)
%    debug: If 1, print figures and intermediate outputs
%
%  [Output]
%    r_C     : distance from the primary body at DV point
%    LowArc  : structure of lower arc parameters
%                - v_Enc  [1x3] velcocity at encounter point
%                - nu_Enc .,   [1x1] true anomaly at encounter point
%                - v_RC   [1x3] velcocity at maneuver point 
%                - nu_RC  [1x1] true anomaly at encounter point
%                - a [1x1] semi-major axis
%                - e [1x1] eccentricity
%                - p [1x1] periapsis
%                - tof [1x1] time of flight
%    HighArc : structure of higher arc parameters (same property as LowArc)
%    F       : number of solutions
%
%  [Reference]
%    - Lantulh, D.V, Russel R.P., and Campagnola S. "Automated Inclution 
%      of V-Infinity Leverging Maneuvers in Gravity-Assist Flyby Tour Design"
%      AAS 12-162
% ************************************************************************


function [r_C,LowArc,HighArc,DV_norm,F] ...
    = vilt_bvp(r_L,r_H,eta_L,theta,tof,N,L,D,C,S,mu,units,debug)

r_L = r_L /units.L;
r_H = r_H /units.L;
tof = tof /units.T;
mu  = mu / (units.L^3/units.T^2);

if debug
    disp(' ');
    disp(['[VILT LAMBERT]  TOF:',num2str(tof,'%.2f'),...
          '  N:',num2str(N,'%d'), '  L:',num2str(L,'%d'), ...
          '  D:',num2str(D,'%d'), '  C:',num2str(N,'%d')]);
end

% etaL = true anomaly of low eccentricity encounter [-pi,pi]
% theta : [0 2pi]
% nu_L  : [-pi, 2*pi] -> [-pi, pi]
% nu_H  : [-3*pi, 3*pi] -> [-pi, pi]

nu_L = eta_L + pi/2*(1-D);
% convert to [-pi,pi]
if nu_L > 2*pi
    error('true anomaly of Low Encounter out of range');
elseif nu_L > pi
    nu_L = - (2*pi - nu_L);
end
    
nu_H = nu_L + C * theta;
% convert to [-pi,pi]
if nu_H > 3*pi
    error('true anomaly of High Encounter out of range');
elseif nu_H > pi
    nu_H = nu_H- 2*pi;
elseif nu_H < (-3*pi)
    error('true anomaly of High Encounter out of range');
elseif nu_H < -pi
    nu_H = nu_H + 2*pi;   
end

if debug
disp(['nu_L = ',num2str(rad2deg(nu_L), '%.0f'),...
      '  theta = ',num2str(rad2deg(theta), '%.0f'),...
      '  nu_H = ',num2str(rad2deg(nu_H), '%.0f')]);
end
  
% initialize solution
r_C = 0; 
DV_norm = 0;
F = 0;
LowArc.v_Enc = []; LowArc.nu_Enc  = []; LowArc.v_RC = []; LowArc.nu_RC = [];
LowArc.a  = []; LowArc.e = []; LowArc.p = []; LowArc.tof = [];
HighArc.v_Enc   = []; HighArc.nu_Enc = []; HighArc.v_RC = []; HighArc.nu_RC = [];
HighArc.a = [];HighArc.e = []; HighArc.p = []; HighArc.tof = [];


%% Determine K_E of EACH ARC
% N = K_L + K_H + 1
if C == 1 % Low -> High
    K_L = L - 1; % first arc
    K_H = N - 1 - K_L;
elseif C == -1 % High -> Low
    K_H = L - 1; % first arc
    K_L = N -1 - K_H;
else
    error('C must be -1 or 1');
end

%% DETERMINE O_L, O_H FROM GEOMETRY AND TRANSFER TYPE

% invalid eta_L
if (eta_L == -pi) || (eta_L == pi)
    F = 0;
    r_C = 0;
    return 
end

[O_L,O_H] = compute_O(nu_L,nu_H);

%% BOUND THE DOMAIN OF INTEREST
if D == 1 % exterior VILT
    rC_LB = max([r_L,r_H]);
    rC_UB = 10;
elseif D == -1 % interior VILT
    eps = 1e-3; % to avoid singlar point
    rC_LB = max([rCsingular(r_H,nu_H), rCsingular(r_L,nu_L)]) + eps;
    rC_UB = min([r_L,r_H]);
    if rC_LB > rC_UB % eps to big
        eps2 = 1e-5;
        rC_LB = rC_UB - eps2;
    end
else
    error('D should be either 1 or -1');
end


%% Print solution step
if debug
    rC_array  = linspace(rC_LB,rC_UB,100);
    TOF_array = zeros(1,length(rC_array));
    for ii = 1:length(rC_array)
        TOF_array(ii) = calcTOF(rC_array(ii), r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);
    end
    figure(1)
    hold on; grid on;
    xlabel('RC (LU)');
    ylabel('TOF (TU)');
    ylim([0 30]);
    plot(rC_array,TOF_array,'r');
    yline(tof,'k--');
end

%% Determine the number of solutions (F) and adjust bounds as needed for S
dTOF_LB = calc_dTOF_dRC(rC_LB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);
tof_LB = calcTOF(rC_LB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);
tof_UB = calcTOF(rC_UB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);

% time of flight not calculatable at boundary
if (isnan(tof_LB)) || (isnan(tof_UB))
    F = 0;
    r_C = 0;
    return
end

if dTOF_LB >= 0   
    if (tof_LB > tof) || (tof_UB < tof) % no solution
        F = 0;
        r_C = 0;
        return
    else
        F = 1;
    end
else
    dTOF_UB = calc_dTOF_dRC(rC_UB, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);
    
    if dTOF_UB >= 0
        % Minimize tof between rC_LB and rC_UB and find rC_min
        optFunc = @(r_C) calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);

        if debug 
            options = optimset('Display','Iter','FunValCheck','on');
            [r_C_min, tof_min] = fminbnd(optFunc,rC_LB,rC_UB,options);
        else
            [r_C_min, tof_min] = fminbnd(optFunc,rC_LB,rC_UB);
        end
        
        if ~isreal(tof_min)
            disp('Minumum tof is imaginary');
        end

        if (tof_min > tof) || (tof_UB < tof) % no solution
            F = 0;
            r_C = 0;
            return
        elseif tof_min == tof
            F = 1;
            r_C = r_C_min;
        elseif tof_LB < tof
            F = 1;
            rC_LB = r_C_min;
        else
            F = 2;
            % adjust the bound to encapture the desired solution
            if S == 1
                rC_UB = r_C_min;
            elseif S == 2
                rC_LB = r_C_min;
            end     
        end

    else % dTOF_UB < 0 -> Not robust regions
        F = 0;
        r_C = 0;
        return
    end
end

if debug 
  disp(['[Bounds]   NSOL: ',num2str(F),'  RC_LB = ',num2str(rC_LB),'  RC_UB = ',num2str(rC_UB), ]);
end

%% SOLVE FOR RC BY NEWTON-RAPHSON
is_solved = 0;
method = 'fzero';
G_UB = calc_G(rC_UB,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H);
G_LB = calc_G(rC_LB,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H);
if (G_LB * G_UB) > 0
   disp('G(UB) and G(LB) has the same value');
end

if strcmp(method, 'Newton-Raphson')
    rC_init = (rC_LB + rC_UB)/2; % initial guess for rC
    [r_C,is_solved] = solveRC(rC_init,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H);
end

% solve using fzero
if strcmp(method, 'fzero') || (~is_solved)
    optFunc = @(r_C) calc_G(r_C,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H);
    if debug
        options = optimset('PlotFcns',@optimplotfval);
        r_C = fzero(optFunc,[rC_LB,rC_UB],options);
    else
        r_C = fzero(optFunc,[rC_LB,rC_UB]);
    end
end

if debug
   disp(['rC = ',num2str(r_C,'%.2f')]);
   disp(' ');
end

%% CALCULATE IN AND OUT VELOCITY AND FLIGHT PATH ANGLE
[v_L, a_L, e_L, p_L] = calcVelocity(r_C,r_L,nu_L);
[v_H, a_H, e_H, p_H] = calcVelocity(r_C,r_H,nu_H);

% invalid situation
% When this happens, trajectory seems to diverge (does not maintain a
% ellipse orbit)
if e_L > e_H
    % disp('  Eccentricity Upside down')   
    F = 0;
    r_C = 0;
   return 
end
%% Result Structure
LowArc.v_Enc     = v_L * units.V;
LowArc.nu_Enc    = nu_L;
LowArc.v_RC      = [0; -D * sqrt(mu/p_L) * (1 + e_L * cos(pi/2 * (1+D))); 0] * units.V;
LowArc.nu_RC     = pi/2 * (D + 1);
LowArc.a         = a_L * units.L;
LowArc.e         = e_L;
LowArc.p         = p_L * units.L;
LowArc.tof       = calcT(r_C, r_L, nu_L, O_L, K_L) * units.T;

HighArc.v_Enc    = v_H * units.V;
HighArc.nu_Enc   = nu_H;
HighArc.v_RC     = [0; -D * sqrt(mu/p_H) * (1 + e_H * cos(pi/2 * (1+D))); 0] * units.V;
HighArc.nu_RC    = pi/2 * (D + 1);
HighArc.a        = a_H * units.L;
HighArc.e        = e_H;
HighArc.p        = p_H * units.L;
HighArc.tof      = calcT(r_C, r_H, nu_H, O_H, K_H) * units.T;

r_C = r_C * units.L;
DV_norm = (norm(LowArc.v_RC) - norm(HighArc.v_RC)) * units.V;

%%%%%% finished function %%%%%%%%%%%%


% ****************************************
%% SUPPLEMENTARY FUNCTIONS
% ****************************************

% return O_L,O_H based on C,D,nu_L,nu_H
function [O_L,O_H] = compute_O(nu_L,nu_H)
   % nu_case
   %   case 1: nu_L < 0 nu_H < 0
   %   case 2: nu_L < 0 nu_H > 0
   %   case 3: nu_L > 0 nu_H < 0
   %   case 4: nu_L > 0 nu_H > 0
   nu_case = compute_nu_case(nu_L,nu_H);
   if D == +1
       if C == +1
          switch nu_case
              case 1
                  O_L = +1;
                  O_H = -1;
              case 2
                  O_L = +1;
                  O_H = +1;                  
              case 3
                  O_L = -1;
                  O_H = -1;                  
              case 4
                  O_L = -1;
                  O_H = +1;                 
          end
       elseif C == -1
          switch nu_case
              case 1
                  O_L = -1;
                  O_H = +1;
              case 2
                  O_L = -1;
                  O_H = -1;                  
              case 3
                  O_L = +1;
                  O_H = +1;                  
              case 4
                  O_L = +1;
                  O_H = -1;                 
          end           
       end
   elseif D == -1
       if C == +1
          switch nu_case
              case 1
                  O_L = -1;
                  O_H = +1;
              case 2
                  O_L = -1;
                  O_H = -1;                  
              case 3
                  O_L = +1;
                  O_H = +1;                  
              case 4
                  O_L = +1;
                  O_H = -1;                 
          end
       elseif C == -1
          switch nu_case
              case 1
                  O_L = +1;
                  O_H = -1;
              case 2
                  O_L = +1;
                  O_H = +1;                  
              case 3
                  O_L = -1;
                  O_H = -1;                  
              case 4
                  O_L = -1;
                  O_H = +1;                 
          end           
       end       
   end
end

% compute nu pattern
function nu_case = compute_nu_case(nu_L,nu_H)
    if (nu_L <= 0) && (nu_H <= 0)
        nu_case = 1;
    elseif (nu_L <= 0) && (nu_H >= 0)
        nu_case = 2;
    elseif (nu_L >= 0) && (nu_H <= 0)
        nu_case = 3;
    elseif (nu_L >= 0) && (nu_H >= 0)
        nu_case = 4;
    end
end

% Calculate radius of singular point
function r_C = rCsingular(r_E, nu_E)
    r_C = r_E/2 * (1 - D * cos(nu_E));
end

% Calculate time of flight for single arc
function t_E = calcT(r_C, r_E, nu_E, O_E, K_E)
    abs_nu_E = abs(nu_E);
%     if abs_nu_E > pi
%         abs_nu_E = 2 * pi - abs_nu_E;
%     end
    
    e_E =                (r_C - r_E)            ...
           /... %------------------------------
                 (r_E*cos(abs_nu_E) + D * r_C);
             
    if (e_E <0) || (e_E > 1)
        t_E = NaN;
        return;
    end
             
    a_E =        r_C * (r_E * cos(abs_nu_E) + D * r_C) ...
          /... %-------------------------------------------
                (r_E * cos(abs_nu_E) + D*(2*r_C - r_E));
            
    E_E = 2 * atan( tan(abs_nu_E/2) * sqrt((1-e_E)/(1+e_E)) ); % [-pi/2, pi/2]
    n_E = sqrt(mu/a_E^3);
    T_E = 2*pi/n_E;
    t_PE = (E_E - e_E * sin(E_E)) / n_E;
    t_E = t_PE * D * O_E + T_E * (K_E + 1/2 - O_E/4 * (D - 1));
end

% Calculate derivative dTOF/dRC
function dTOF_dRC = calc_dTOF_dRC(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H)
    dTL_dRC = calc_dTE_dRC(r_C,r_L,nu_L,O_L,K_L);
    dTH_dRC = calc_dTE_dRC(r_C,r_H,nu_H,O_H,K_H);
    dTOF_dRC = dTL_dRC + dTH_dRC;
end

% Calculate derivative dTE/dRC
% Generated equation using symbolic math toolbox
% -pi < nu_E < pi
function dTE_dRC = calc_dTE_dRC(r_C,r_E,nu_E,O_E,K_E)
  nu_E = abs(nu_E);
  
  dTE_dRC = ...
     (pi*((3*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^4*(D*r_C + r_E*cos(nu_E))^3) ...
     - (6*D*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^2)/(r_C^3*(D*r_C + r_E*cos(nu_E))^3) ...
     + (3*D*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^3*(D*r_C + r_E*cos(nu_E))^4)) ...
     *(K_E - (O_E*(D - 1))/4 + 1/2))/((mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3) ...
     /(r_C^3*(r_C*D + r_E*cos(nu_E))^3))^(3/2) - D*O_E*(sin(2*atan(tan(nu_E/2) ...
     *(-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1)/((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1))^(1/2))) ...
     /((D*r_C + r_E*cos(nu_E))*((mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^3*(r_C*D + r_E*cos(nu_E))^3))^(1/2)) ...
     - (tan(nu_E/2)*((1/(D*r_C + r_E*cos(nu_E)) - (D*(r_C - r_E))/(D*r_C + r_E*cos(nu_E))^2)/((r_C - r_E) ...
     /(D*r_C + r_E*cos(nu_E)) + 1) - ((1/(D*r_C + r_E*cos(nu_E)) ...
     - (D*(r_C - r_E))/(D*r_C + r_E*cos(nu_E))^2)*((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1)) ...
     /((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1)^2))/((-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1) ...
     /((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1))^(1/2)*((tan(nu_E/2)^2*((r_C - r_E) ...
     /(D*r_C + r_E*cos(nu_E)) - 1))/((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1) - 1)) ...
     - (D*sin(2*atan(tan(nu_E/2)*(-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1) ...
     /((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1))^(1/2)))*(r_C - r_E)) ...
     /((D*r_C + r_E*cos(nu_E))^2*((mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3) ...
     /(r_C^3*(r_C*D + r_E*cos(nu_E))^3))^(1/2)) + (sin(2*atan(tan(nu_E/2) ...
     *(-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1)/((r_C - r_E) ...
     /(D*r_C + r_E*cos(nu_E)) + 1))^(1/2)))*(r_C - r_E) ...
     *((3*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^4*(D*r_C + r_E*cos(nu_E))^3) ...
     - (6*D*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^2)/(r_C^3*(D*r_C + r_E*cos(nu_E))^3) ...
     + (3*D*mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^3*(D*r_C + r_E*cos(nu_E))^4))) ...
     /(2*(D*r_C + r_E*cos(nu_E))*((mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3) ...
     /(r_C^3*(r_C*D + r_E*cos(nu_E))^3))^(3/2)) + (tan(nu_E/2)*cos(2*atan(tan(nu_E/2) ...
     *(-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1)/((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1))^(1/2))) ...
     *((1/(D*r_C + r_E*cos(nu_E)) - (D*(r_C - r_E))/(D*r_C + r_E*cos(nu_E))^2)/((r_C - r_E) ...
     /(D*r_C + r_E*cos(nu_E)) + 1) - ((1/(D*r_C + r_E*cos(nu_E)) - (D*(r_C - r_E)) ...
     /(D*r_C + r_E*cos(nu_E))^2)*((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1)) ...
     /((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1)^2)*(r_C - r_E)) ...
     /((D*r_C + r_E*cos(nu_E))*(-((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) - 1) ...
     /((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1))^(1/2)*((tan(nu_E/2)^2*((r_C - r_E) ...
     /(D*r_C + r_E*cos(nu_E)) - 1))/((r_C - r_E)/(D*r_C + r_E*cos(nu_E)) + 1) - 1) ...
     *((mu*(r_E*cos(nu_E) + D*(2*r_C - r_E))^3)/(r_C^3*(r_C*D + r_E*cos(nu_E))^3))^(1/2)));
 
end

% Caculate total time of flight

function tof = calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H)
   t_L = calcT(r_C, r_L, nu_L, O_L, K_L);
   t_H = calcT(r_C, r_H, nu_H, O_H, K_H);
   
   if (~isnan(t_L)) && (~isnan(t_H))
       tof = t_L + t_H;
   else
       tof = NaN;
   end
end

% Solve RC using newton-raphson
function [r_C,is_solved] = solveRC(rC_init, tof, r_L, r_H,...
                                   nu_L, nu_H, O_L, O_H, K_L, K_H)
    tol = 1e-10;
    maxIterN = 100;
    is_solved = 0; 
    alpha = 0.5; % step coefficient
    
    r_C = rC_init;
    
    if debug
        disp('  [Newton-Raphson]');
    end
    
    for i = 1:maxIterN
        G = calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H) - tof;
        dG_dRC = calc_dTOF_dRC(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H);
        dRC = G/dG_dRC;
        r_C_new = r_C - alpha * dRC;
        
        rate = alpha;
        while r_C_new < 0
            rate = rate * alpha;
            r_C_new = r_C - rate*dRC;
        end
        r_C = r_C_new;
        
        if debug 
            disp(['    i: ',num2str(i),' rC = ',num2str(r_C,'%.4f'),'  |G| = ',...
                num2str(abs(G),'%.3e'),'  dR_C= ',...
                num2str(dRC,'%.3e')]);
        end
        
        if abs(G) < tol
            if debug 
              disp('  Converged!!');
            end
            is_solved = 1;
            break;
        end
    end    
end

% Calculate difference between TOF estimate and calculated TOF
function G = calc_G(r_C,tof,r_L,r_H,nu_L,nu_H,O_L,O_H,K_L,K_H)
   G = calcTOF(r_C, r_L, r_H, nu_L, nu_H, O_L, O_H, K_L, K_H) - tof;
end

% caclulate Velocity in PQW frame

function [v_E, a_E, e_E, p_E] = calcVelocity(r_C,r_E,nu_E)
   r_hat = [cos(nu_E); sin(nu_E); 0];
   v_hat = [-sin(nu_E);cos(nu_E); 0];
   a_E = r_C * (r_E * cos(nu_E) + D * r_C) / (r_E * cos(nu_E) ...
       + D*(2*r_C - r_E));
   e_E = (r_C - r_E)/(r_E*cos(nu_E) + D * r_C);
   p_E = a_E * (1 - e_E^2);
   rdot_Norm  =  sqrt(mu/p_E) * e_E * sin(nu_E);
   rvdot_Norm =  sqrt(mu/p_E) * (1 + e_E * cos(nu_E));
   v_E = rdot_Norm * r_hat + rvdot_Norm * v_hat;
end



end
