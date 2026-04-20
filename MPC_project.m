%% MPC tracking using SCP nominal trajectory
% Compatible with saved SCP file: scp_nominal_trajectory.mat
% Assumes traj contains:
%   traj.X_nom   = 7xN nominal states
%   traj.U_nom   = 2xN nominal controls from SCP, where U = [Tbx; Tbz]
%   traj.dt, traj.tf, traj.N, traj.t, traj.X0, traj.Xf, traj.p

clear; clc; close all;

%% Load SCP nominal trajectory
load('scp_nominal_trajectory.mat','traj');

Xnom = traj.X_nom;      % 7 x N
Uscp = traj.U_nom;      % 2 x N, SCP control = [Tbx; Tbz]
dt   = traj.dt;
tf   = traj.tf;
N    = traj.N;
t    = traj.t;
p    = traj.p;
X0   = traj.X0;
Xf   = traj.Xf; 

m0 = Xnom(7,1);

%% Convert SCP control [Tbx; Tbz] -> MPC control [T; delta]
Tnom_full = sqrt(Uscp(1,:).^2 + Uscp(2,:).^2);
dnom_full = atan2(Uscp(1,:), Uscp(2,:));

% Use k = 1,...,N-1 for control sequence
Unom = [Tnom_full(1:N-1);
        dnom_full(1:N-1)];

%% MPC design
nx = 6;   % regulate first 6 states
nu = 2;

Np = 20;  % prediction horizon

Q  = diag([20, 5, 50, 10, 300, 30]);
R  = diag([1e-6, 10]);
% Qf = diag([5000, 5000, 5000, 5000, 20000, 8000]);
Qf = 20*Q;

%% Constraints
Tmin = 0;
Tmax = p.T_max;
deltamax = p.delta_max;

%% Closed-loop simulation
X = zeros(7,N);
U = zeros(2,N-1);

% Initial condition = nominal + perturbation
X(:,1) = Xnom(:,1) + [2; -1; 3; -1; deg2rad(3); deg2rad(0.5); 0];

% opts = optimoptions('quadprog','Display','off');
opts = optimset('Display','off');
for k = 1:N-1

    % Horizon shrinks near the end
    Nh = min(Np, N-k);

    % Build lifted prediction model
    Abar = zeros(nx*Nh, nx);
    Bbar = zeros(nx*Nh, nu*Nh);
    Qbar = zeros(nx*Nh, nx*Nh);
    Rbar = zeros(nu*Nh, nu*Nh);

    Ad_list = zeros(nx,nx,Nh);
    Bd_list = zeros(nx,nu,Nh);

    %% Linearize and discretize along nominal over prediction horizon
    for i = 1:Nh
        idx = k + i - 1;

        Xk = Xnom(:,idx);
        Uk = Unom(:,idx);   % [T; delta]

        [~, A, B] = rocketEOM(Xk, Uk, p);

        A6 = A(1:6,1:6);
        B6 = B(1:6,:);

        % Exact ZOH discretization
        M  = [A6, B6;
              zeros(nu,nx+nu)];
        Md = expm(M*dt);

        Ad_list(:,:,i) = Md(1:nx,1:nx);
        Bd_list(:,:,i) = Md(1:nx,nx+1:nx+nu);
    end

    %% Build prediction matrices
    for i = 1:Nh
        % State transition from current deviation dx0 to future deviation dxi
        Phi = eye(nx);
        for j = 1:i
            Phi = Ad_list(:,:,j) * Phi;
        end
        Abar((i-1)*nx+1:i*nx, :) = Phi;

        % Input-to-state blocks
        for j = 1:i
            Gamma = Bd_list(:,:,j);
            for ell = j+1:i
                Gamma = Ad_list(:,:,ell) * Gamma;
            end
            Bbar((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = Gamma;
        end

        % Cost blocks
        if i < Nh
            Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx) = Q;
        else
            Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx) = Qf;
        end

        Rbar((i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu) = R;
    end

    %% Current state deviation from nominal
    dx0 = X(1:6,k) - Xnom(1:6,k);

    %% QP cost:
    % minimize (Abar*dx0 + Bbar*dU)' Qbar (Abar*dx0 + Bbar*dU) + dU' Rbar dU
    H = 2*(Bbar' * Qbar * Bbar + Rbar);
    f = 2*(Bbar' * Qbar * Abar * dx0);

    % Symmetrize H for numerical safety
    H = 0.5*(H + H');

    %% Input bounds on deviation dU = U - Unom
    lb = zeros(nu*Nh,1);
    ub = zeros(nu*Nh,1);

    for i = 1:Nh
        idx = k + i - 1;
        unom_i = Unom(:,idx);

        lb((i-1)*nu+1:i*nu) = [Tmin - unom_i(1);
                              -deltamax - unom_i(2)];

        ub((i-1)*nu+1:i*nu) = [Tmax - unom_i(1);
                               deltamax - unom_i(2)];
    end

    %% Optional state deviation bounds
    % Can be added later as Aineq*dU <= bineq if needed

    %% Solve QP
    
    dUopt = quadprog(H, f, [], [], [], [], lb, ub, [], opts);

    if isempty(dUopt)
        % Fallback to nominal if solver fails
        uk = Unom(:,k);
    else
        du0 = dUopt(1:nu);
        uk = Unom(:,k) + du0;
    end

    % Saturation safeguard
    uk(1) = min(max(uk(1), Tmin), Tmax);
    uk(2) = min(max(uk(2), -deltamax), deltamax);

    U(:,k) = uk;

    %% Nonlinear propagation
    xdot = rocketEOM(X(:,k), uk, p);
    X(:,k+1) = X(:,k) + dt*xdot;

    % Optional mass floor
    if X(7,k+1) <= 1000
        X(7,k+1) = 1000;
    end
end

%% Convert applied MPC inputs back to SCP-style body components, if needed
Tbx_cl = U(1,:) .* sin(U(2,:));
Tbz_cl = U(1,:) .* cos(U(2,:));

%% Save MPC results
mpc_result.X = X;
mpc_result.U = U;                % [T; delta]
mpc_result.Tbx = Tbx_cl;
mpc_result.Tbz = Tbz_cl;
mpc_result.t = t;
mpc_result.Xnom = Xnom;
mpc_result.Unom = Unom;
save('mpc_tracking_result.mat','mpc_result');

%% Plots
figure('Name','MPC Tracking');

subplot(3,2,1);
plot(t, Xnom(1,:), '--', t, X(1,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('x [m]');
legend('Nominal','MPC');
title('Horizontal Position');

subplot(3,2,2);
plot(t, Xnom(3,:), '--', t, X(3,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('z [m]');
legend('Nominal','MPC');
title('Altitude');

subplot(3,2,3);
plot(t, Xnom(2,:), '--', t, X(2,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('v_x [m/s]');
legend('Nominal','MPC');
title('Horizontal Velocity');

subplot(3,2,4);
plot(t, Xnom(4,:), '--', t, X(4,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('v_z [m/s]');
legend('Nominal','MPC');
title('Vertical Velocity');

subplot(3,2,5);
plot(t, rad2deg(Xnom(5,:)), '--', t, rad2deg(X(5,:)), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('\theta [deg]');
legend('Nominal','MPC');
title('Pitch Angle');

subplot(3,2,6);
plot(t, Xnom(7,:), '--', t, X(7,:), 'LineWidth', 1.5);
grid on; xlabel('Time [s]'); ylabel('Mass [kg]');
legend('Nominal','MPC');
title('Mass');

figure('Name','MPC Control');
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
legend('Nominal','MPC');
title('Flight Path');
axis equal;

%% ------------------------------------------------------------------------
% Nonlinear rocket dynamics + Jacobians for control [T; delta]
% State: X = [x; vx; z; vz; theta; omega; m]
% Control: U = [T; delta]
%% ------------------------------------------------------------------------
function [X_dot, A, B] = rocketEOM(X, U, p)

    % States
    x  = X(1); %#ok<NASGU>
    vx = X(2);
    z  = X(3); %#ok<NASGU>
    vz = X(4);
    th = X(5);
    w  = X(6);
    m  = X(7);

    % Controls
    T     = U(1);
    delta = U(2);

    % Nonlinear dynamics
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
        % A = df/dx
        A = [0 1 0 0 0 0 0;
             0 0 0 0 (T/m)*cos(th+delta) 0 -(T/m^2)*sin(th+delta);
             0 0 0 1 0 0 0;
             0 0 0 0 -(T/m)*sin(th+delta) 0 -(T/m^2)*cos(th+delta);
             0 0 0 0 0 1 0;
             0 0 0 0 0 0 (6*T/(m^2*p.L))*sin(delta);
             0 0 0 0 0 0 0];

        % B = df/du = [df/dT, df/ddelta]
        B = [0, 0;
             (1/m)*sin(th+delta),  (T/m)*cos(th+delta);
             0, 0;
             (1/m)*cos(th+delta), -(T/m)*sin(th+delta);
             0, 0;
             -(6/(m*p.L))*sin(delta), -(6*T/(m*p.L))*cos(delta);
             -1/(p.Isp*p.g), 0];
    end
end