% Falcon 9 trajectory generation (sequential convex programming)
clear
%% initialization
p.g     = 9.81;
p.L     = 41.2;
p.Isp   = 285;
p.T_max = 845000 * 3;          % 3 Merlin 1D engines
p.delta_max = deg2rad(5);      % 5 degree gimbal limit
m0      = 30000;               % Initial mass (kg)

% Initial Conditions: [x, vx, z, vz, theta, omega, m]
X0 = [500; -50; 1000; -200; deg2rad(10); -0.01; m0];
Xf = [0; 0; 0; 0; 0; 0];       % Target state 

% Discretization
N = 101;                       
tf = 10.0;                     
dt = tf / (N-1);

%% Reference trajectory
% straight line to the launch pad
X_ref = zeros(7, N);
for i = 1:7
    X_ref(i,:) = linspace(X0(i), (i<=6)*Xf(min(i,6)) + (i==7)*m0*0.8, N);
end
U_ref = repmat([p.T_max*0.5; 0], 1, N);   % [T; delta=0] initially

%% SCP loop
max_iters = 10;
for iter = 1:max_iters
    fprintf('SCP Iteration %d / %d... ', iter, max_iters);
    
    % Define YALMIP Decision Variables
    X   = sdpvar(7, N, 'full');
    U   = sdpvar(2, N, 'full');  
    nu  = sdpvar(7, N-1, 'full'); % virtual control
    
    Constraints = [];
    
    % BCs
    Constraints = [Constraints, X(:,1) == X0];
    Constraints = [Constraints, X(1:6, N) == Xf(1:6)];
    
    % Trust Region scaling for different units [x, vx, z, vz, theta, omega, m]
    TR = [1000; 200; 1000; 200; pi/2; pi/2; 10000] / iter;
    
    % Dynamics/ Path Constraints
    for k = 1:N-1
        [F_bar, A_rk4, B_rk4] = get_rk4_linearization(X_ref(:,k), U_ref(:,k), dt, p);
        X_next = F_bar + A_rk4*(X(:,k)-X_ref(:,k)) + B_rk4*(U(:,k)-U_ref(:,k)) + nu(:,k);
        Constraints = [Constraints, X(:,k+1) == X_next];
        
        % Trust Regions 
        Constraints = [Constraints, -TR <= X(:,k) - X_ref(:,k) <= TR];
        
        
        % Gimbal Angle Limits (Linearized cone constraint)
        Constraints = [Constraints, 0 <= U(1,k) <= p.T_max];
        Constraints = [Constraints, -p.delta_max <= U(2,k) <= p.delta_max];

        % State Bounds (Keep the rocket pointing generally UP)
        Constraints = [Constraints, -deg2rad(30) <= X(5,k) <= deg2rad(30)]; % Max pitch 30 deg
        Constraints = [Constraints, -deg2rad(40) <= X(6,k) <= deg2rad(40)]; % Max rotation rate
    end
    
    % Apply constraints to the final node
    Constraints = [Constraints, 0 <= U(1,N) <= p.T_max];
    Constraints = [Constraints, -p.delta_max <= U(2,N) <= p.delta_max];
    
    % Penalize Virtual Controls (Cheating)
    nu_penalty = sum(sum(abs(nu)));
    
    % Penalize Control Chatter (Normalized to T_max so it doesn't break the solver)
    delta_U_norm = (U(:, 2:N) - U(:, 1:N-1)) / p.T_max;
    chatter_penalty = sum(sum(delta_U_norm.^2)); 
    
    % Maximize mass, penalize Gam slightly, MASSIVELY penalize cheating, gently penalize chatter
    Objective = -X(7,N) + 1e6 * nu_penalty + 50 * chatter_penalty;

    %solve
    options = sdpsettings('solver', 'mosek', 'verbose', 0);
    sol = optimize(Constraints, Objective, options);
    % Guard against bad solves — don't update reference if solution is invalid
    X_new   = value(X);
    U_new   = value(U);
    
    if any(isnan(X_new(:))) || any(isnan(U_new(:))) || sol.problem ~= 0
        fprintf('  WARNING: Solver failed (code %d), keeping previous reference\n', sol.problem);
    else
        X_ref   = X_new;
        U_ref   = U_new;
    end
    
    cheat_val = value(nu_penalty);
    fprintf('Solved. Virtual Control Usage: %.10f\n', cheat_val);
    if cheat_val < 1e-3 && iter > 2
       break;
    end
end

fprintf('\nOptimization Complete! Final Mass: %.1f kg\n', X_ref(7,end));

%% Plots
t_vec = linspace(0, tf, N);

figure('Name', 'Convex Optimization Landing Trajectory');
subplot(2,2,1);
plot(X_ref(1,:), X_ref(3,:), 'b-', 'LineWidth', 2); hold on;
plot(X_ref(1,1), X_ref(3,1), 'go', 'MarkerSize', 8, 'LineWidth', 2);
plot(X_ref(1,end), X_ref(3,end), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Downrange x (m)'); ylabel('Altitude z (m)');
title('Flight Profile'); grid on; axis equal;

subplot(2,2,2);
plot(t_vec, X_ref(5,:) * 180/pi, 'k-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Pitch Angle (deg)');
title('Attitude'); grid on;

subplot(2,2,3);
T_mag     = U_ref(1,:);
plot(t_vec, T_mag / 1000, 'r-', 'LineWidth', 2); hold on;
yline(p.T_max / 1000, 'r--', 'T_{max}');
xlabel('Time (s)'); ylabel('Thrust (kN)');
title('Thrust Profile'); grid on;

subplot(2,2,4);
delta_deg = U_ref(2,:) * 180/pi;
plot(t_vec, delta_deg, 'b-', 'LineWidth', 2); hold on;
yline(rad2deg(p.delta_max), 'b--', 'Limit');
yline(-rad2deg(p.delta_max), 'b--', 'Limit');
xlabel('Time (s)'); ylabel('Gimbal Angle (deg)');
title('Gimbal Profile'); grid on;

% Replace get_linear_dyn and the Euler integration block with this

function [F_bar, A_rk4, B_rk4] = get_rk4_linearization(X_bar, U_bar, dt, p)
    % Evaluate RK4 at the linearization point (nonlinear, exact)
    k1 = nonlinear_f(X_bar,        U_bar, p);
    k2 = nonlinear_f(X_bar+dt/2*k1, U_bar, p);
    k3 = nonlinear_f(X_bar+dt/2*k2, U_bar, p);
    k4 = nonlinear_f(X_bar+dt*k3,   U_bar, p);
    F_bar = X_bar + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);  % nominal next state

    % Linearize the RK4 map via chain rule
    [~, Ak1] = nonlinear_f_jac(X_bar,         U_bar, p);
    [~, Ak2] = nonlinear_f_jac(X_bar+dt/2*k1, U_bar, p);
    [~, Ak3] = nonlinear_f_jac(X_bar+dt/2*k2, U_bar, p);
    [~, Ak4] = nonlinear_f_jac(X_bar+dt*k3,   U_bar, p);

    % STM via RK4 chain rule: dF/dX
    A_rk4 = eye(7) + (dt/6)*(Ak1 ...
                            + 2*Ak2*(eye(7) + dt/2*Ak1) ...
                            + 2*Ak3*(eye(7) + dt/2*Ak2) ...
                            + Ak4  *(eye(7) + dt*  Ak3));

    % dF/dU via RK4 chain rule
    [~, ~, Bk1] = nonlinear_f_jac(X_bar,         U_bar, p);
    [~, ~, Bk2] = nonlinear_f_jac(X_bar+dt/2*k1, U_bar, p);
    [~, ~, Bk3] = nonlinear_f_jac(X_bar+dt/2*k2, U_bar, p);
    [~, ~, Bk4] = nonlinear_f_jac(X_bar+dt*k3,   U_bar, p);

    % Note: B also has direct terms at each stage since U enters f directly
    % Full expression:
    B_rk4 = (dt/6)*(Bk1 + 2*Bk2 + 2*Bk3 + Bk4);   % simplified: U constant across stages
end

function f = nonlinear_f(X, U, p)
    % Your EOM.m exactly
    vx=X(2); vz=X(4); th=X(5); om=X(6); m=X(7);
    T=U(1); delta=U(2);
    f = [vx;
         (T/m)*sin(th+delta);
         vz;
         (T/m)*cos(th+delta) - p.g;
         om;
         -(6*T/(m*p.L))*sin(delta);
         -T/(p.Isp*p.g)];
end

function [f, A, B] = nonlinear_f_jac(X, U, p)
    vx=X(2); vz=X(4); th=X(5); om=X(6); m=X(7);
    T=U(1); delta=U(2);
    f = nonlinear_f(X, U, p);
    A = zeros(7,7);
    A(1,2)=1; A(3,4)=1; A(5,6)=1;
    A(2,5)=(T/m)*cos(th+delta);  A(2,7)=-(T/m^2)*sin(th+delta);
    A(4,5)=-(T/m)*sin(th+delta); A(4,7)=-(T/m^2)*cos(th+delta);
    A(6,7)=(6*T/(m^2*p.L))*sin(delta);
    B = zeros(7,2);
    B(2,1)=(1/m)*sin(th+delta);   B(2,2)=(T/m)*cos(th+delta);
    B(4,1)=(1/m)*cos(th+delta);   B(4,2)=-(T/m)*sin(th+delta);
    B(6,1)=-(6/(m*p.L))*sin(delta); B(6,2)=-(6*T/(m*p.L))*cos(delta);
    B(7,1)=-1/(p.Isp*p.g);
end
save('X_ref.mat', 'X_ref');
save('U_ref.mat', 'U_ref');   % already [T; delta]