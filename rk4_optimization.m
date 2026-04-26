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
X0 = [500; -50; 1000; -200; deg2rad(10); 0; m0];
Xf = [0; 0; 0; 0; 0; 0];

% Discretization
N = 100;                       
tf = 10.0;                     
dt = tf / (N-1);

%% Reference trajectory
X_ref = zeros(7, N);
for i = 1:7
    X_ref(i,:) = linspace(X0(i), (i<=6)*Xf(min(i,6)) + (i==7)*m0*0.8, N);
end
U_ref   = repmat([0; m0 * p.g], 1, N); 
Gam_ref = repmat(m0 * p.g, 1, N);      

%% SCP loop
max_iters = 10;
for iter = 1:max_iters
    fprintf('SCP Iteration %d / %d... ', iter, max_iters);
    
    X   = sdpvar(7, N, 'full');
    U   = sdpvar(2, N, 'full');  
    Gam = sdpvar(1, N, 'full');  
    nu  = sdpvar(7, N-1, 'full');
    
    Constraints = [];
    
    Constraints = [Constraints, X(:,1) == X0];
    Constraints = [Constraints, X(1:6, N) == Xf(1:6)];
    
    TR = [1000; 200; 1000; 200; pi/2; pi/2; 10000] / iter;
    
    for k = 1:N-1
        [f_bar, A, B, C] = get_linear_dyn(X_ref(:,k), U_ref(:,k), Gam_ref(k), p);
        
        % RK4 Integration WITH Virtual Control (nu)
        % k1
        f1 = f_bar + A*(X(:,k) - X_ref(:,k)) + B*(U(:,k) - U_ref(:,k)) + C*(Gam(k) - Gam_ref(k));
        y1 = X(:,k) + f1 * dt/2;
        % k2
        f2 = f_bar + A*(y1 - X_ref(:,k)) + B*(U(:,k) - U_ref(:,k)) + C*(Gam(k) - Gam_ref(k));
        y2 = X(:,k) + f2 * dt/2;
        % k3
        f3 = f_bar + A*(y2 - X_ref(:,k)) + B*(U(:,k) - U_ref(:,k)) + C*(Gam(k) - Gam_ref(k));
        y3 = X(:,k) + f3 * dt;
        % k4
        f4 = f_bar + A*(y3 - X_ref(:,k)) + B*(U(:,k) - U_ref(:,k)) + C*(Gam(k) - Gam_ref(k));
        
        X_next = X(:,k) + (f1 + 2*f2 + 2*f3 + f4) * dt/6 + nu(:,k);
        Constraints = [Constraints, X(:,k+1) == X_next];
        
        Constraints = [Constraints, -TR <= X(:,k) - X_ref(:,k) <= TR];
        
        Constraints = [Constraints, norm(U(:,k)) <= Gam(k)];
        Constraints = [Constraints, Gam(k) <= p.T_max];
        
        Constraints = [Constraints,  U(1,k) <= U(2,k) * tan(p.delta_max)];
        Constraints = [Constraints, -U(1,k) <= U(2,k) * tan(p.delta_max)];
        Constraints = [Constraints,  U(2,k) >= 0]; 

        Constraints = [Constraints, -deg2rad(30) <= X(5,k) <= deg2rad(30)];
        Constraints = [Constraints, -deg2rad(40) <= X(6,k) <= deg2rad(40)];
    end
    
    % Final node constraints
    Constraints = [Constraints, norm(U(:,N)) <= Gam(N)];
    Constraints = [Constraints, Gam(N) <= p.T_max];
    Constraints = [Constraints,  U(1,N) <= U(2,N) * tan(p.delta_max)];
    Constraints = [Constraints, -U(1,N) <= U(2,N) * tan(p.delta_max)];
    Constraints = [Constraints,  U(2,N) >= 0];
    
    nu_penalty      = sum(sum(abs(nu)));
    delta_U_norm    = (U(:, 2:N) - U(:, 1:N-1)) / p.T_max;
    chatter_penalty = sum(sum(delta_U_norm.^2)); 
    
    Objective = -X(7,N) + 1e-4 * sum(Gam) + 1e6 * nu_penalty + 50 * chatter_penalty;

    options = sdpsettings('solver', 'mosek', 'verbose', 0);
    sol = optimize(Constraints, Objective, options);
    
    cheat_val = value(nu_penalty);
    fprintf('Solved. Virtual Control Usage: %.4f\n', cheat_val);
    
    X_ref   = value(X);
    U_ref   = value(U);
    Gam_ref = value(Gam);

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
T_mag = sqrt(U_ref(1,:).^2 + U_ref(2,:).^2);
plot(t_vec, T_mag / 1000, 'r-', 'LineWidth', 2); hold on;
yline(p.T_max / 1000, 'r--', 'T_{max}');
xlabel('Time (s)'); ylabel('Thrust (kN)');
title('Thrust Profile'); grid on;

subplot(2,2,4);
delta_deg = atan2(U_ref(1,:), U_ref(2,:)) * 180/pi;
plot(t_vec, delta_deg, 'b-', 'LineWidth', 2); hold on;
yline(rad2deg(p.delta_max), 'b--', 'Limit');
yline(-rad2deg(p.delta_max), 'b--', 'Limit');
xlabel('Time (s)'); ylabel('Gimbal Angle (deg)');
title('Gimbal Profile'); grid on;

%% EOM
function X_dot = eom(~, x0, U, p)
    vx = x0(2); vz = x0(4); th = x0(5); w = x0(6); m = x0(7);
    Tbx = U(1); Tbz = U(2);
    X_dot = [vx;
             (Tbx*cos(th) + Tbz*sin(th))/m;
             vz;
             (-Tbx*sin(th) + Tbz*cos(th))/m - p.g;
             w;
             -(6/p.L)*(Tbx/m);
             -norm(U)/(p.Isp*p.g)];
end

%% RK4 Integration
function yk1 = RK4(t, X_true, dt, U, p)
    k1 = eom(t,         X_true,              U, p);
    y1 = X_true + k1 * dt/2;
    k2 = eom(t + dt/2,  y1,                  U, p);
    y2 = X_true + k2 * dt/2;
    k3 = eom(t + dt/2,  y2,                  U, p);
    y3 = X_true + k3 * dt;
    k4 = eom(t + dt,    y3,                  U, p);
    yk1 = X_true + (k1 + 2*k2 + 2*k3 + k4) * dt/6;
end

%% Linearized Dynamics
function [f, A, B, C] = get_linear_dyn(X, U, Gam, p)
    vx = X(2); z = X(3); vz = X(4); th = X(5); w = X(6); m = X(7);
    Tbx = U(1); Tbz = U(2);
    
    f = [vx;
         (Tbx*cos(th) + Tbz*sin(th))/m;
         vz;
         (-Tbx*sin(th) + Tbz*cos(th))/m - p.g;
         w;
         -(6/p.L)*(Tbx/m);
         -Gam/(p.Isp*p.g)];
         
    A = zeros(7,7);
    A(1,2) = 1;
    A(2,5) = (-Tbx*sin(th) + Tbz*cos(th))/m;
    A(2,7) = -(Tbx*cos(th) + Tbz*sin(th))/(m^2);
    A(3,4) = 1;
    A(4,5) = (-Tbx*cos(th) - Tbz*sin(th))/m;
    A(4,7) = -(-Tbx*sin(th) + Tbz*cos(th))/(m^2);
    A(5,6) = 1;
    A(6,7) = (6/p.L)*(Tbx/(m^2));
    
    B = zeros(7,2);
    B(2,1) =  cos(th)/m;   B(2,2) = sin(th)/m;
    B(4,1) = -sin(th)/m;   B(4,2) = cos(th)/m;
    B(6,1) = -(6/p.L)/m;   B(6,2) = 0;
    
    C = zeros(7,1);
    C(7,1) = -1/(p.Isp*p.g);
end