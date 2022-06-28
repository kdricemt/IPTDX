% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function [t,y] = twobody(y0,tspan,mu)
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
%{
  y             - a matrix whose 12 columns are, respectively,
                  X1,Y1,Z1; X2,Y2,Z2; VX1,VY1,VZ1; VX2,VY2,VZ2
  XG,YG,ZG      - column vectors containing the X,Y and Z coordinates (km)
                  the center of mass at the times in t
  User M-function required:   rkf45
  User subfunctions required: rates, output
%}
% ----------------------------------------------------------------------

%...Integrate the equations of motion:
% options = odeset('RelTol',1e-10,'AbsTol',1e-15,'Refine',6);
% [t,sv] = ode113(@rates,tspan,y0,options); 

y0 = [y0(1:3);y0(4:6)];
options = odeset('RelTol',1e-13,'AbsTol',1e-22,'Refine',6);
[t,y]   = ode113(@rates,tspan,y0,options);

% ~~~~~~~~~~~~~~~~~~~~~~~~
function dydt = rates(t,y)
% ~~~~~~~~~~~~~~~~~~~~~~~~
%{
  This function calculates the accelerations in Equations 2.19
  t     - time
  y     - column vector containing the position and velocity vectors
           of the system at time t
  r     - position vector
  v     - velocity vector

  R     - magnitude of the position vector

  dydt  - column vector containing the velocity and acceleration
           vectors of the system at time t
%}
% ------------------------
r    = [y(1); y(2); y(3)];
v    = [y(4); y(5); y(6)];

R    = norm(r);
    
a    = -mu*r/R^3;

dydt = [v;a;];

end %rates
% ~~~~~~~~~~~~~~~~~~

end

