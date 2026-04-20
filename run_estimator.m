clear
addpath('position_filters')
load("X_ref.mat")
load("U_ref.mat")

rng(42)

% Process Noise: Let the filter trust the measurements more during aggressive maneuvers
Q = diag([5^2, 5^2, 5^2, 5^2, 0.05^2, 0.05^2, 10^2]);

R = diag([1^2, 0.01^2, 0.01^2, 0.5^2]);  % relax mass measurement trust less

const.sample_rate = 10;
const.g = 9.81;
const.I_sp = 285;
const.L = 41.2;

tf = 10;
% t = linspace(0, tf, tf * const.sample_rate);
t_span = 0 : (1/const.sample_rate) : tf;

% Get reference trajectory and control inputs

t_ref = linspace(0, tf, size(U_ref, 2)); % The discrete time vector from SCP

T_ref = sqrt(U_ref(1,:).^2 + U_ref(2,:).^2);
delta_ref = atan2(U_ref(1,:), U_ref(2,:));

X_0 = X_ref(:,1);

% Setting uncertainty to roughly: 10m pos, 2m/s vel, 0.1rad pitch, 100kg mass
P_0 = diag([10^2, 2^2, 10^2, 2^2, 0.1^2, 0.01^2, 100^2]);

options = odeset('RelTol', 1E-12,'AbsTol', 1E-12);
[t, X_truth] = ode45(@(t, X) true_dynamics_wrapper(t, X, t_ref, T_ref, delta_ref, const), t_span, X_0, options);

% Start with perfect state and covariance knowledge
X_est = X_0;
P_est = P_0;

X_est_hist = zeros(length(t), 7);
P_est_hist = zeros(7, 7, length(t));

% attitude variables initialization
theta0 = X_ref(5, 1);
wz0    = X_ref(6, 1);
m0     = X_ref(7, 1);

q0_init = cos(theta0 / 2);
q3_init = sin(theta0 / 2);

% Attitude State: [q0, q1, q2, q3, wx, wy, wz, m]
X_att_est = [q0_init; 0; 0; q3_init; 0; 0; wz0; m0];
P_att_est = diag([0.01^2 * ones(1,4), 0.01^2 * ones(1,3), 100^2]);

% attitude noise matrices
Q_att = diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1e-1]);
R_att = diag([0.01^2 * ones(1,4), 0.05^2 * ones(1,3), 0.01^2]);

for i = 1:length(t)
    U.T = interp1(t_ref, T_ref, t(i), 'linear', 'extrap');
    U.delta = interp1(t_ref, delta_ref, t(i), 'linear', 'extrap');

    % Get measurements
    z_truth = h(X_truth(i,:)');
    % z = z_truth + 0.01 * randn(4, 1);
    noise_range = 1.0 * randn();
    noise_angle = 0.01 * randn();
    noise_theta = 0.01 * randn();  % (This gets overwritten by attitude EKF anyway)
    noise_mass  = 0.5 * randn();
    
    z = z_truth + [noise_range; noise_angle; noise_theta; noise_mass];

    % dummy noisy sensor measurement for attitude ekf
    z_att = map_truth2sensors(X_truth(i,:));
    
    % Run ekf
    [X_att_est, P_att_est, theta_est] = attitude_EKF_function(X_att_est, P_att_est, z_att, Q_att, R_att, U, const);

    z(3) = theta_est;

    %[X_est, P_est] = position_EKF(X_est, P_est, z, Q, R, U, const);
    R_dynamic = R;
    if X_est(3) < 20  % If the rocket is below 20 meters altitude
        R_dynamic(2,2) = 1e6;  % Inflate angle variance so the filter ignores it
    end
    
    % Pass the dynamic R matrix instead of the static one
    [X_est, P_est] = position_UKF(X_est, P_est, z, Q, R_dynamic, U, const);

    X_est_hist(i,:) = X_est;
    P_est_hist(:,:,i) = P_est;
    fprintf('t=%.1f  theta_truth=%.4f  theta_est=%.4f\n', t(i), X_truth(i,5), theta_est);
end

%% Plotting with Correct Labels
state_labels = {'Downrange x (m)', 'Velocity v_x (m/s)', 'Altitude z (m)', ...
                'Velocity v_z (m/s)', 'Pitch \theta (rad)', 'Pitch Rate \omega_z (rad/s)', 'Mass m (kg)'};

% Figure 1: Error Plots
figure(1)
set(gcf, 'Name', 'Estimation Errors with 3-Sigma Bounds', 'Position', [100, 100, 800, 800])
sgtitle('State Estimation Error (Estimate - Truth)')
for i = 1:7
    subplot(7, 1, i)
    hold on; grid on;
    % Plot Error
    plot(t, X_est_hist(:,i) - X_truth(:,i), 'k', 'LineWidth', 1.2, 'DisplayName', 'Error')
    % Plot 3-Sigma Bounds
    plot(t, 3 * sqrt(squeeze(P_est_hist(i, i, :))), 'r--', 'LineWidth', 1, 'DisplayName', '+3\sigma')
    plot(t, -3 * sqrt(squeeze(P_est_hist(i, i, :))), 'r--', 'LineWidth', 1, 'DisplayName', '-3\sigma')
    
    ylabel(state_labels{i})
    if i == 7
        xlabel('Time (s)')
    end
    if i == 1
        legend('Location', 'best')
    end
end
    
% Figure 2: True vs Estimated
figure(2)
set(gcf, 'Name', 'True vs Estimated States', 'Position', [950, 100, 800, 800])
sgtitle('True vs Estimated Rocket States')
for i = 1:7
    subplot(7, 1, i)
    hold on; grid on;
    plot(t, X_truth(:,i), 'b', 'LineWidth', 1.5, 'DisplayName', 'Truth')
    plot(t, X_est_hist(:,i), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimate')
    
    ylabel(state_labels{i})
    if i == 7
        xlabel('Time (s)')
    end
    if i == 1
        legend('Location', 'best')
    end
end

% function dX = dummy_dynamics(t, X, U, const)
% % Dummy dynamics with sinusoidal control inputs
% I_sp = const.I_sp;
% g = const.g;
% L = const.L;
% 
% T = abs(U.T * sin(t));
% delta = U.delta * sin(t);
% 
% x = X(1);
% dx = X(2);
% z = X(3);
% dz = X(4);
% theta = X(5);
% dtheta = X(6);
% m = X(7);
% 
% dX = [dx;
%       T / m * sin(theta + delta);
%       dz;
%       T / m * cos(theta + delta) - g;
%       dtheta;
%       -6 * T / (m * L) * sin(delta);
%       -T / (I_sp * g)];
% end

function z_meas = h(X)
% Map the state to a measurement
% z(1): radar distance measurement
% z(2): radar angle measurement
% z(3): attitude angle measurement
% z(4): mass measurement

x = X(1);
z = X(3);
theta = X(5);
m = X(7);

z_meas = [sqrt(x^2 + z^2);
          -atan2(x, z) - theta;
          theta;
          m];
end

function z_att = map_truth2sensors(X_truth)
    % extract variables
    theta_true = X_truth(5);
    wz_true    = X_truth(6);
    m_true     = X_truth(7);

    % convert pitch angle to quaternion
    q0_true = cos(theta_true / 2);
    q1_true = 0;
    q2_true = 0;
    q3_true = sin(theta_true / 2);

    % assemble array for attitude EKF
    z_att_truth = [q0_true; q1_true; q2_true; q3_true; 0; 0; wz_true; m_true];

    % add noise
    noise_q = 0.01 * randn(4,1);
    noise_w = 0.05 * randn(3,1);
    noise_m = 0.01 * randn(1,1);
    
    z_att = z_att_truth + [noise_q; noise_w; noise_m];

    z_att(1:4) = z_att(1:4) / norm(z_att(1:4));
end

function dX = true_dynamics_wrapper(t, X, t_ref, T_ref, delta_ref, const)
    % 1. Calculate the dynamic inputs for THIS exact millisecond
    U_dynamic.T = interp1(t_ref, T_ref, t, 'linear', 'extrap');
    U_dynamic.delta = interp1(t_ref, delta_ref, t, 'linear', 'extrap');
    
    % 2. Call your universal EOM function. 
    % We only ask for the first output (X_dot) because ode45 doesn't need the Jacobian.
    X_dot = EOM(t, X, U_dynamic, const);
    
    dX = X_dot;
end