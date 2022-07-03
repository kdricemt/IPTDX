%% Plot VILT in PQW frame

function plot_VILT_PQW(mu, C, D, r_L, r_H, r_C, LowArc, HighArc, figureNum)
    nu_L = LowArc.nu_Enc;
    nu_H = HighArc.nu_Enc;
    
    if D == 1
        domain = 'exterior';
    elseif D == -1
        domain = 'interior';
    end

    if C == 1 % Low -> high
        y0_arc1   = [r_L*cos(nu_L); r_L*sin(nu_L); 0; LowArc.v_Enc];
        yend_arc2 = [r_H*cos(nu_H); r_H*sin(nu_H); 0; HighArc.v_Enc];
        % D=+1 apoapsis, D = -1 periapsis
        y0_arc2 = [-D * r_C; 0; 0; HighArc.v_RC];
        tof_arc1 = LowArc.tof;
        tof_arc2 = HighArc.tof;
        arcLabel1 = 'Low';
        arcLabel2 = 'High';
    elseif C == -1 % High -> Low
        y0_arc1 = [r_H*cos(nu_H);r_H*sin(nu_H); 0; HighArc.v_Enc];
        yend_arc2 = [r_L*cos(nu_L); r_L*sin(nu_L); 0; LowArc.v_Enc];
        % D=+1 apoapsis, D = -1 periapsis
        y0_arc2 = [-D * r_C; 0; 0; LowArc.v_RC];         
        tof_arc1 = HighArc.tof;
        tof_arc2 = LowArc.tof;
        arcLabel1 = 'High';
        arcLabel2 = 'Low';        
    end
    
    % propagate trajectory
    [tarc1,yarc1]  = twobody(y0_arc1,[0 tof_arc1],mu); 
    [tarc2,yarc2]  = twobody(y0_arc2,[0 tof_arc2],mu); 
    
    %% plot
    figure(figureNum);
    hold on; grid on; axis equal;
    xlabel('x');
    ylabel('y');
    
    % plot center planet
    r = 0.02;
    phi = deg2rad(1:360);
    plot(r*cos(phi), r*sin(phi), 'k', 'DisplayName', 'Center');

    plot(r_L*cos(phi), r_L*sin(phi), 'k--', 'DisplayName', 'r_L');
    plot(r_H*cos(phi), r_H*sin(phi), 'k--', 'DisplayName', 'r_H');
    
    
    title(['VILT Trajectory (',domain,') ',arcLabel1, ' -> ',arcLabel2,...
        '    \nu_L = ',num2str(rad2deg(nu_L), '%.0f'),' ^\circ',...
        '  \nu_H = ',num2str(rad2deg(nu_H), '%.0f'),' ^\circ']);
    plot(yarc1(1,1), yarc1(1,2),'ro','DisplayName','Start');
    text(yarc1(1,1) + 0.1, yarc1(1,2),'Start Point');
    plot(yend_arc2(1), yend_arc2(2),'bo', 'DisplayName','End');
    text(yend_arc2(1) + 0.1, yend_arc2(2),'End Point');
    plot(-D * r_C, 0,'go', 'DisplayName','Maneuver');
    text(-D * r_C + 0.1, 0, 'Maneuver Point');    
    plot(yarc1(:,1), yarc1(:,2), 'r', 'DisplayName','1st arc');
    plot(yarc2(:,1), yarc2(:,2), 'b','DisplayName','2nd arc');
    legend
end


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

