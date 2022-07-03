%% Test VILT-BVP software

clear all; close all;
addpath("../src")

simCase = 6;   % change here

switch simCase
    case 1 %a
        mu = 1; D = 1; C = 1; N = 2; L = 1; S = 1; 
        r_L   = 0.9000; 
        r_H   = 0.9018;
        theta = 6.0751;
        tof   = 18.68;
        eta_L = 0.00;
    case 2 %b
        mu = 1; D = 1; C = 1; N = 2; L = 1; S = 2; 
        r_L   = 0.9000; 
        r_H   = 1.0502;
        theta = 4.1016;
        tof   = 16.839;
        eta_L = 0.00;
    case 3 %c
        mu = 1; D = -1; C = 1; N = 3; L = 2; S = 2; 
        r_L   = 0.9000; 
        r_H   = 0.9000;
        theta = 6.2600;
        tof   = 12.548;
        eta_L = 0.00;   
    case 4 %d
        mu = 1; D = -1; C = 1; N = 3; L = 1; S = 2; 
        r_L   = 0.9000; 
        r_H   = 0.9026;
        theta = 6.0290;
        tof   = 12.359;
        eta_L = 0.00;   
    case 5 %e
        mu = 1; D = 1; C = 1; N = 2; L = 1; S = 1; 
        r_L   = 0.9000; 
        r_H   = 0.9002;
        theta = 6.2446;
        tof   = 18.818;
        eta_L = -0.7854; 
    case 6 %f
        mu = 1; D = 1; C = 1; N = 2; L = 1; S = 2; 
        r_L   = 0.9000; 
        r_H   = 0.9924;
        theta = 4.6883;
        tof   = 17.455;
        eta_L = -0.7854;  
end

% Search for Lambert -------------------------------------
debug = 1;
units.L = 1; units.T = 1; units.V = 1; % units for non-dimentionalization
[r_C,LowArc,HighArc,DV_norm,F] ...
    = vilt_bvp(r_L,r_H,eta_L,theta,tof,N,L,D,C,S,mu,units,debug);
disp(['rC= ',num2str(r_C, '%.3f'), ...
      '  DV Norm= ', num2str(DV_norm, '%.3e')]);
plot_VILT_PQW(mu, C, D, r_L, r_H, r_C, LowArc, HighArc, 3)
