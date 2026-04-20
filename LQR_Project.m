

clear; clc; close all;

%% Loading SCP nominal trajectory
load('scp_nominal_trajectory.mat','traj');

Xnom = traj.X_nom;      
Uscp = traj.U_nom;      %  SCP control = [Tbx; Tbz]
dt   = traj.dt;
tf   = traj.tf;
N    = traj.N;
t    = traj.t;
p    = traj.p;
X0   = traj.X0;
Xf   = traj.Xf; 

% Use initial mass from nominal
m0 = Xnom(7,1);

%% Convert SCP control [Tbx; Tbz] -> LQR control [T; delta]
% T     = thrust magnitude
% delta = gimbal angle
Tnom_full = sqrt(Uscp(1,:).^2 + Uscp(2,:).^2);
dnom_full = atan2(Uscp(1,:), Uscp(2,:));

% LQR propagation/control uses k = 1,...,N-1
Unom = [Tnom_full(1:N-1);
        dnom_full(1:N-1)];

%% LQR weights on deviation states [x; vx; z; vz; theta; omega]
Q  = diag([20, 5, 50, 10, 3000, 30]);
% Q  = diag([20, 5, 50, 10, 500, 30]);
R  = diag([1e-6, 10]);
Qf = 20*Q;

%% Preallocate linearized/discretized models
nx = 6;   % regulate first 6 states; mass propagated separately
nu = 2;

Ad = zeros(nx,nx,N-1);
Bd = zeros(nx,nu,N-1);

%% Linearize and discretize along nominal trajectory
for k = 1:N-1
    Xk = Xnom(:,k);
    Uk = Unom(:,k);   % [T; delta]

    [~, A, B] = rocketEOM(Xk, Uk, p);

    A6 = A(1:6,1:6);
    B6 = B(1:6,:);

    % Exact ZOH discretization using matrix exponential
    M = [A6, B6;
         zeros(nu,nx+nu)];
    Md = expm(M*dt);

    Ad(:,:,k) = Md(1:nx,1:nx);
    Bd(:,:,k) = Md(1:nx,nx+1:nx+nu);
end

%% Backward Riccati recursion for TV-LQR
K = zeros(nu,nx,N-1);
P = Qf;

for k = N-1:-1:1
    Ak = Ad(:,:,k);
    Bk = Bd(:,:,k);

    K(:,:,k) = (R + Bk'*P*Bk) \ (Bk'*P*Ak);
    P = Q + Ak'*P*Ak - Ak'*P*Bk*K(:,:,k);
end

%% Closed-loop nonlinear simulation
X = zeros(7,N);
U = zeros(2,N-1);

% Start near nominal with a perturbation
X(:,1) = Xnom(:,1) + [2; -1; 3; -1; deg2rad(3); deg2rad(0.5); 0];

% Input limits from SCP parameters
% Tmin = 0.3 * p.T_max;
Tmin = 0;
% Tmin = 0.001 * p.T_max;   % try 20% first
Tmax = p.T_max;
deltamax = p.delta_max;

for k = 1:N-1
    % Deviation from nominal on regulated states
    dx = X(1:6,k) - Xnom(1:6,k);

    % TV-LQR tracking law
    uk = Unom(:,k) - K(:,:,k)*dx;

    % Saturation
    uk(1) = min(max(uk(1), Tmin), Tmax);
    uk(2) = min(max(uk(2), -deltamax), deltamax);

    U(:,k) = uk;

    % Nonlinear propagation
    xdot = rocketEOM(X(:,k), uk, p);
    X(:,k+1) = X(:,k) + dt*xdot;

    % Optional safety on mass
    if X(7,k+1) <= 1000
        X(7,k+1) = 1000;
    end
end

%% Convert applied LQR inputs back to SCP-style body components, if needed
Tbx_cl = U(1,:) .* sin(U(2,:));
Tbz_cl = U(1,:) .* cos(U(2,:));

%% Save LQR results
lqr_result.X = X;
lqr_result.U = U;                  % [T; delta]
lqr_result.Tbx = Tbx_cl;
lqr_result.Tbz = Tbz_cl;
lqr_result.K = K;
lqr_result.t = t;
lqr_result.Xnom = Xnom;
lqr_result.Unom = Unom;
save('lqr_tracking_result.mat','lqr_result');

%% Plots
figure('Name','TV-LQR Tracking');

subplot(3,2,1);
plot(t, Xnom(1,:), '--', t, X(1,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('x [m]');
legend('Nominal','Closed-loop');
title('Horizontal Position');

subplot(3,2,2);
plot(t, Xnom(3,:), '--', t, X(3,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('z [m]');
legend('Nominal','Closed-loop');
title('Altitude');

subplot(3,2,3);
plot(t, Xnom(2,:), '--', t, X(2,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('v_x [m/s]');
legend('Nominal','Closed-loop');
title('Horizontal Velocity');

subplot(3,2,4);
plot(t, Xnom(4,:), '--', t, X(4,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('v_z [m/s]');
legend('Nominal','Closed-loop');
title('Vertical Velocity');

subplot(3,2,5);
plot(t, rad2deg(Xnom(5,:)), '--', t, rad2deg(X(5,:)), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('\theta [deg]');
legend('Nominal','Closed-loop');
title('Pitch Angle');

subplot(3,2,6);
plot(t, Xnom(7,:), '--', t, X(7,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('Mass [kg]');
legend('Nominal','Closed-loop');
title('Mass');

figure('Name','TV-LQR Control');
subplot(2,1,1);
plot(t(1:end-1), Unom(1,:), '--', t(1:end-1), U(1,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('T [N]');
legend('Nominal','Applied');
title('Thrust Magnitude');

subplot(2,1,2);
plot(t(1:end-1), rad2deg(Unom(2,:)), '--', t(1:end-1), rad2deg(U(2,:)), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('\delta [deg]');
legend('Nominal','Applied');
title('Gimbal Angle');

figure('Name','Trajectory in x-z Plane');
plot(Xnom(1,:), Xnom(3,:), '--', 'LineWidth', 1.5); hold on;
plot(X(1,:), X(3,:), 'LineWidth', 1.5);
grid on; xlabel('x [m]'); ylabel('z [m]');
legend('Nominal','Closed-loop');
title('Flight Path');
axis equal;

%% ------------------------------------------------------------------------
% Nonlinear rocket dynamics + Jacobians for control [T; delta]
% State: X = [x; vx; z; vz; theta; omega; m]
% Control: U = [T; delta]
%% ------------------------------------------------------------------------
function [X_dot, A, B] = rocketEOM(X, U, p)

    % States
    x     = X(1); %#ok<NASGU>
    vx    = X(2);
    z     = X(3); %#ok<NASGU>
    vz    = X(4);
    th    = X(5);
    w     = X(6);
    m     = X(7);

    % Controls
    T     = U(1);
    delta = U(2);

    % Dynamics
    x_ddot     = (T/m) * sin(th + delta);
    z_ddot     = (T/m) * cos(th + delta) - p.g;
    theta_ddot = -(6*T/(m*p.L)) * sin(delta);
    m_dot      = -T/(p.Isp*p.g);

    X_dot = [vx;
             x_ddot;
             vz;
             z_ddot;
             w;
             theta_ddot;
             m_dot];

    if nargout > 1
        % Jacobian A = df/dx
        A = [0 1 0 0 0 0 0;
             0 0 0 0 (T/m)*cos(th+delta) 0 -(T/m^2)*sin(th+delta);
             0 0 0 1 0 0 0;
             0 0 0 0 -(T/m)*sin(th+delta) 0 -(T/m^2)*cos(th+delta);
             0 0 0 0 0 1 0;
             0 0 0 0 0 0 (6*T/(m^2*p.L))*sin(delta);
             0 0 0 0 0 0 0];

        % Jacobian B = df/du = [df/dT, df/ddelta]
        B = [0, 0;
             (1/m)*sin(th+delta),  (T/m)*cos(th+delta);
             0, 0;
             (1/m)*cos(th+delta), -(T/m)*sin(th+delta);
             0, 0;
             -(6/(m*p.L))*sin(delta), -(6*T/(m*p.L))*cos(delta);
             -1/(p.Isp*p.g), 0];
    end
end