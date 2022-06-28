function [r_L,r_H,theta] = calc_theta(sv1,sv2,C)
    % Calculate 
    if C == -1 % High -> Low
        r_L = norm(sv2(1:3));
        r_H = norm(sv1(1:3));
        sv1_unit = sv1(1:3)/r_H;
        sv2_unit = sv2(1:3)/r_L;
    elseif C == 1 %low -> High
        r_L = norm(sv1(1:3));
        r_H = norm(sv2(1:3));           
        sv1_unit = sv1(1:3)/r_L;
        sv2_unit = sv2(1:3)/r_H;        
    end
    % calculate sv1 -> sv2 [0 2*pi]
    theta = acos(sv1_unit'*sv2_unit); % [0, pi] 
    cross_12 = cross(sv1_unit,sv2_unit);
    
    % print for debug
    % disp(['calc theta: ',num2str(rad2deg(theta)), '  cross(3): ', num2str(cross_12(3),'%.4f')]);
    
    if cross_12(3) < 0 % z < 0: clock wise -> change to counter clock wise
        theta = 2*pi - theta;
    end

end