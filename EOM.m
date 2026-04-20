% This script defines a symbolic function for the dynamic equations of
% motion for the rigid, constant density rod that is approximating the
% rocket. 

function [X_dot, F] = EOM(t, X, U, const)
    % Unpack State
    x=X(1); x_dot = X(2); z = X(3); z_dot = X(4);
    theta = X(5); theta_dot = X(6); m = X(7);
    
    %Unpack inputs
    T = U.T; delta = U.delta;
    
    %constants
    g = const.g;
    L = const.L;
    I_sp = const.I_sp;
    
    
    x_ddot = T/m * sin(theta+delta);
    
    z_ddot = T/m * cos(theta+delta) - g;
    
    theta_ddot = -6*T/(m*L) * sin(delta);
    
    m_dot = -T/(I_sp * g); 
    
    X_dot = [x_dot; x_ddot; z_dot; z_ddot; theta_dot; theta_ddot; m_dot];
    
    % Jacobian of state vector
    
    F = [0 1 0 0 0 0 0; 
         0 0 0 0 (T/m)*cos(theta+delta) 0 -(T/m^2)*sin(theta+delta);
         0 0 0 1 0 0 0;
         0 0 0 0 -(T/m)*sin(theta+delta) 0 -(T/m^2)*cos(theta+delta);
         0 0 0 0 0 1 0;
         0 0 0 0 0 0 (6*T/(m^2*L))*sin(delta);
         zeros(1,7)];
end

