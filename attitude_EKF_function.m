function [X_att_est, P_att_est, theta_est] = attitude_EKF_function(X_att_prev, P_att_prev, z_att, Q_att, R_att, U, const)
    dt = 1 / const.sample_rate;
    
    U_vec = [U.T; U.delta];
    
    %% Prediction
    X_pred = RK4(0, X_att_prev, dt, U_vec);
    X_pred(1:4) = X_pred(1:4) / norm(X_pred(1:4)); % normalize quaternion
    
    % get jacobian and STM for propagation
    F = get_Jacobian(X_pred, U_vec);
    Phi = expm(F * dt); 
    
    % Update error covariance matrix
    P_pred = Phi * P_att_prev * Phi' + Q_att;
    
    %% Correction
    K = P_pred / (P_pred + R_att); % (assume H = eye(8)
    
    % Innovation 
    y = z_att - X_pred;
    
    % update state estimate
    X_att_est = X_pred + K * y;
    X_att_est(1:4) = X_att_est(1:4) / norm(X_att_est(1:4)); % Normalize quaternion
    
    % update covariance matrix
    P_att_est = (eye(8) - K) * P_pred;
    
    % extract theta
    % Convert the updated Z-axis rotation quaternion back into a pitch angle
    q0 = X_att_est(1);
    q3 = X_att_est(4);
    theta_est = 2 * atan2(q3, q0);
    
end

% rk4 integrator
function yk1 = RK4(t, X_true, dt, U)
    % Notice I removed m from inputs here, because eom extracts it directly!
    k1 = eom(t, X_true, U);
    y1 = X_true + k1*dt/2;
    k2 = eom(t+dt/2, y1, U);
    y2 = X_true + k2*(dt/2);
    k3 = eom(t+dt/2, y2, U);
    y3 = X_true + k3*(dt);
    k4 = eom(t+dt, y3, U);
    yk1 = X_true + (k1 + 2*k2 + 2*k3 + k4)*dt / 6;
end

% Jacobian
function F = get_Jacobian(X, U)
   L = 41.2;
   q0 = X(1); q1 = X(2); q2 = X(3); q3 = X(4); wx = X(5); wy = X(6); wz = X(7); m = X(8);
   T = U(1); delta = U(2); 
   Omega = [0 -0  -0  -wz; 0  0   wz -0; 0 -wz  0   0; wz 0  -0   0]; 
   Ksi = [-q1 -q2 -q3; q0 -q3 q2; q3 q0 -q1; -q2 q1 q0];
   F = [0.5*Omega, 0.5*Ksi, zeros(4,1);
        zeros(2,8);
        zeros(1,7), ((6*T)/(m^2 * L))*sin(delta);
        zeros(1,8)]; 
end

%eom

function X_dot = eom(~, x0, U)
    L = 41.2; I_sp = 285; g = 9.81;
    q0 = x0(1); q1 = x0(2); q2 = x0(3); q3 = x0(4); wx = x0(5); wy = x0(6); wz = x0(7); m = x0(8);
    T = U(1); delta = U(2);
    Omega = [0 -0  -0  -wz; 0  0   wz -0; 0 -wz  0   0; wz 0  -0   0];  
    q = [q0; q1; q2; q3];
    q_dot = 0.5 * Omega * q;
    dwz = -6*T/(m*L) * sin(delta);
    dw = [0; 0; dwz];
    m_dot = -T/(I_sp * g);
    X_dot = [q_dot; dw; m_dot];
end