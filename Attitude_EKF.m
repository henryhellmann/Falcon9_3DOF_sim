% This script defines the EKF for attitude estimation

%% Initialization
dt = 0.01;     % integration time step 
Tf = 20;       % final time of simulation
g = 9.81;
L = 41.2;
m = 433100; % kg (wet mass)

%Jacobian - symbolic
%state variables
syms q0 q1 q2 q3 wx wy wz
X = [q0 q1 q2 q3 wx wy wz]';   % state vector 

%EoMs
dw = [0;0;0];                  % initialized angular rate Eom

Omega = [0 -wx -wy -wz; 
         wx 0 wz -wy; 
         wy -wz 0 wx; 
         wz wy -wx 0];

q = [q0;q1;q2;q3];

q_dot = 0.5 * Omega * q;

%Jacobian
Ksi = [-q1 -q2 -q3; 
        q0 -q3 q2; 
        q3 q0 -q1; 
       -q2 q1 q0];
w = X(5:end);

F = [0.5*Omega, 0.5*Ksi; 
    zeros(3,7)];          % Jacobian matrix
% Convert it into a callable numerical function
function F = get_Jacobian(X,U)
   L = 41.2;
   % unpack X = [q0 q1 q2 q3 wx wy wz m]'
   q0 = X(1); q1 = X(2); q2 = X(3); q3 = X(4); wx = X(5); wy = X(6); wz = X(7); m = X(8);
   
   % unpack U
   T = U(1); delta = U(2); 

   Omega = [0 -0  -0  -wz; 
             0  0   wz -0; 
             0 -wz  0   0; 
             wz 0  -0   0]; 
    %Jacobian
    Ksi = [-q1 -q2 -q3; 
            q0 -q3 q2; 
            q3 q0 -q1; 
           -q2 q1 q0];
    F = [0.5*Omega, 0.5*Ksi, zeros(4,1);
        zeros(2,8);
        zeros(1,7), ((6*T)/m^2 * L)*sin(delta);
        zeros(1,8)];          % Jacobian matrix
end

% Eom Function
function X_dot = eom(~,x0, U)
    % constants
    L = 41.2; % meters
    I_sp = 285; % seconds
    g = 9.81;

    % unpack X = [q0 q1 q2 q3 wx wy wz m]'
    q0 = x0(1); q1 = x0(2); q2 = x0(3); q3 = x0(4); wx = x0(5); wy = x0(6); wz = x0(7); m = x0(8);

    % unpack control input
    T = U(1); delta = U(2);

    % quaternion kinematics
    Omega = [0 -0  -0  -wz; 
             0  0   wz -0; 
             0 -wz  0   0; 
             wz 0  -0   0];  
    q = [q0;q1;q2;q3];
    q_dot = 0.5 * Omega * q;
    
    % Euler rotational dynamics
    dwz = -6*T/(m*L) * sin(delta);
    
    dw = [0;0;dwz];

    m_dot = -T/(I_sp * g);
    
    X_dot = [q_dot;dw; m_dot];
end

%%
% Define variances of quaternion and gyroscope noise here ******
var_q = 0.01^2;             % initial value, change later
var_w = 0.05^2;             % (rad/s)^2 initial value, change later
var_m = 0.01^2;

% Filter initialization
X0 = [1;1;1;1;1;1;1;1];       % initial 8x1 state estimate guess
P0 = eye(8) * 1e-6;         % Error Covariance matrix, assume perfect initial knowledge at low values, increase for testing convergence
Q0 = [eye(4)*1e-8, zeros(4,4);
      zeros(4,4), diag([1e-6 1e-6 1e-6 1e-7])]; % Process Noise Cov matrix, how much do you trust mathematical model   
R0 = diag([var_q var_q var_q var_q var_w var_w var_w var_m]);  %measurement noise cov matrix, based on variance of noise I inject

% DataLogging empty arrays
q_true = zeros(4,Tf/dt);

%% Simulation loop

options = odeset('RelTol',1E-10,'AbsTol',1E-10);

% angle input
delta = 0;

t = linspace(0,Tf,Tf/dt);

q_init = [0.9962; 0; 0; 0.0872]; % initial quaternion
q_init = q_init/norm(q_init); %normalize

w_init = [0;0;0];

X_est = zeros(8,length(t));
X_est(:,1) = [q_init; zeros(3,1); m];
P_current = P0;
P_history = zeros(8,length(t));

% Set initial condition for state
X_true = [zeros(7,length(t));m, zeros(1,length(t)-1)];
X_true(1:7,1) = [q_init; w_init];
X_noisy = [zeros(7,length(t));m, zeros(1,length(t)-1)];
X_noisy(:,1) = X_true(:,1); % perfect first measurement


for i = 1:length(t)-1
    % get true mass and define reference control input
    delta = 0.05 * sin(2 * pi * 0.5 * t(i));
    m_curr = X_true(8,i);
    T = m_curr*g;

    U = [T;delta];

    % integrate true forward
    X_true(:,i+1) = RK4(t(i), X_true(:,i),dt, U);
    X_true(1:4, i+1) = X_true(1:4, i+1)/norm(X_true(1:4, i+1)); % Normalize q
    
    % Generate noise
    noise_q = sqrt(var_q) * randn(4,1);
    noise_w = sqrt(var_w) * randn(3,1);
    noise_m = sqrt(var_m) * randn(1,1);
    
    % add noise to state
    X_noisy(:, i+1) = X_true(:, i+1) + [noise_q; noise_w; noise_m];
    X_noisy(1:4, i+1) = X_noisy(1:4, i+1) / norm(X_noisy(1:4, i+1));

    % Prediction step
    X_est(:,i+1) = RK4(t(i), X_est(:,i),dt, U);
    X_est(1:4,i+1) = X_est(1:4,i+1) / norm(X_est(1:4,i+1)); % normalize
    
    % call jacobian
    F = get_Jacobian(X_est(:,i+1), U);
    Phi = expm(F*dt); %Convert to discrete STM
    
    % Update error covarianve matrix P
    P_pred = Phi*P_current*Phi' + Q0;

    % Correction step
    % Kalman Gain update
    K = P_pred * (P_pred + R0)^(-1);

    % Innovation
    y = X_noisy(:,i+1) - X_est(:,i+1);

    %Update state estimate
    X_est(:,i+1) = X_est(:,i+1) + K*y;
    X_est(1:4,i+1) = X_est(1:4,i+1)/norm(X_est(1:4,i+1)); % normalize

    % Update covariance matrix
    P_current = (eye(8) - K)*P_pred;
    P_history(:,i+1) = diag(P_current);
end

%% RK4 integration Function
function yk1 = RK4(t,X_true,dt, U)
    k1 = eom(t,X_true, U);
    y1 = X_true + k1*dt/2;

    k2 = eom(t+dt/2,y1, U);
    y2 = X_true + k2*(dt/2);

    k3 = eom(t+dt/2,y2, U);
    y3 = X_true + k3*(dt);

    k4 = eom(t+dt,y3, U);
    yk1 = X_true + (k1 + 2*k2 + 2*k3 + k4)*dt / 6;
end



%% Plotting

% 1. Plot Quaternions (True vs. Estimated vs. Noisy)
figure('Name', 'Quaternion Estimation', 'Position', [100, 100, 800, 600]);
sgtitle('Quaternion State Estimation');
for k = 1:4
    subplot(2, 2, k);
    hold on; grid on;
    % Plot noisy measurements (light gray so it doesn't overpower the plot)
    plot(t, X_noisy(k, :), 'Color', [0.8 0.8 0.8], 'DisplayName', 'Noisy Measurement');
    % Plot True State
    plot(t, X_true(k, :), 'b', 'LineWidth', 1.5, 'DisplayName', 'True');
    % Plot Estimated State
    plot(t, X_est(k, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimate');
    
    ylabel(['q_', num2str(k-1)]);
    xlabel('Time (s)');
    if k == 1
        legend('Location', 'best');
    end
end

% 2. Plot Angular Velocities (True vs. Estimated vs. Noisy)
figure('Name', 'Angular Velocity Estimation', 'Position', [150, 150, 800, 600]);
sgtitle('Angular Velocity Estimation (\omega)');
axis_labels = {'\omega_x (rad/s)', '\omega_y (rad/s)', '\omega_z (rad/s)'};
for k = 1:3
    subplot(3, 1, k);
    hold on; grid on;
    plot(t, X_noisy(k+4, :), 'Color', [0.8 0.8 0.8], 'DisplayName', 'Noisy Measurement');
    plot(t, X_true(k+4, :), 'b', 'LineWidth', 1.5, 'DisplayName', 'True');
    plot(t, X_est(k+4, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimate');
    
    ylabel(axis_labels{k});
    xlabel('Time (s)');
    if k == 1
        legend('Location', 'best');
    end
end

% 3. Plot Estimation Error with 3-Sigma Bounds
% Note: This requires P_history to be logged in your loop!
if exist('P_history', 'var')
    figure('Name', 'Estimation Error', 'Position', [200, 200, 900, 700]);
    sgtitle('State Estimation Error with 3\sigma Bounds');
    
    % Calculate Error (Estimate - True)
    error = X_est - X_true;
    
    % Calculate 3-sigma bounds from the logged covariance diagonals
    sigma3 = 3 * sqrt(P_history);
    
    % Plot Quaternion Errors
    for k = 1:4
        subplot(4, 2, 2*k-1);
        hold on; grid on;
        plot(t, error(k, :), 'k', 'LineWidth', 1.2);
        plot(t, sigma3(k, :), 'r--', 'LineWidth', 1);
        plot(t, -sigma3(k, :), 'r--', 'LineWidth', 1);
        ylabel(['e_{q', num2str(k-1), '}']);
        if k == 4, xlabel('Time (s)'); end
        if k == 1, title('Quaternion Error'); end
    end
    
    % Plot Angular Velocity Errors
    for k = 1:3
        subplot(4, 2, 2*k);
        hold on; grid on;
        plot(t, error(k+4, :), 'k', 'LineWidth', 1.2);
        plot(t, sigma3(k+4, :), 'r--', 'LineWidth', 1);
        plot(t, -sigma3(k+4, :), 'r--', 'LineWidth', 1);
        ylabel(['e_{\omega', num2str(k), '}']);
        if k == 3, xlabel('Time (s)'); end
        if k == 1, title('Angular Velocity Error'); end
    end
else
    disp('Warning: Add P_history to your loop to plot the 3-sigma error bounds.');
end
