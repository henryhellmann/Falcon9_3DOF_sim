clc; clear;

p.g = 9.81;
p.L = 41.2;
p.Isp = 285;
p.T_max = 845000*3;
p.delta_max = deg2rad(5);
m0 = 30000;

X0 = [500; -50; 1000; -200; deg2rad(10); 0; m0];
Xf = [0; 0; 0; 0; 0; 0];

N = 100;                       
tf = 35.0;                     
dt = tf / (N-1);

gamma_min = deg2rad(3);

X_ref = zeros(7, N);
for i = 1:6
    X_ref(i,:) = linspace(X0(i), Xf(i), N);
end
X_ref(7,:) = linspace(m0, m0 * 0.8, N);

T_hover = m0 * p.g;
U_ref   = repmat([0; T_hover], 1, N);
Gam_ref = repmat(T_hover * 1.1, 1, N);

max_iters = 10;
for iter = 1:max_iters
    fprintf('SCP Iteration %d / %d... ', iter, max_iters);

    X   = sdpvar(7, N, 'full');
    U   = sdpvar(2, N, 'full');  
    Gam = sdpvar(1, N, 'full');  
    nu  = sdpvar(7, N-1, 'full');
    
    Constraints = [];
    
    Constraints = [Constraints, X(:,1) == X0];
    Constraints = [Constraints, X(1,N) == Xf(1)];   
    Constraints = [Constraints, X(2,N) == Xf(2)];   
    Constraints = [Constraints, X(3,N) == Xf(3)];   
    Constraints = [Constraints, X(4,N) == Xf(4)];  
    Constraints = [Constraints, X(5,N) == Xf(5)]; 
    Constraints = [Constraints, X(6,N) == Xf(6)]; 

    Constraints = [Constraints, X(3,:) >= 0];

    TR = [1000; 200; 1000; 200; pi/2; pi/2; 10000] / iter;
    
    for k = 1:N-1
        [f_bar, A, B, C] = get_linear_dyn(X_ref(:,k), U_ref(:,k), Gam_ref(k), p);
        
        X_next = X(:,k) + dt * (f_bar + A*(X(:,k) - X_ref(:,k)) ...
                                      + B*(U(:,k) - U_ref(:,k)) ...
                                      + C*(Gam(k) - Gam_ref(k))) + nu(:,k);
        Constraints = [Constraints, X(:,k+1) == X_next];
        Constraints = [Constraints, -TR <= X(:,k) - X_ref(:,k) <= TR];        
        Constraints = [Constraints, norm(U(:,k)) <= Gam(k)];
        Constraints = [Constraints, Gam(k) <= p.T_max];
        Constraints = [Constraints,  U(1,k) <= U(2,k) * tan(p.delta_max)];
        Constraints = [Constraints, -U(1,k) <= U(2,k) * tan(p.delta_max)];
        Constraints = [Constraints,  U(2,k) >= 0];

        Constraints = [Constraints, X(3,k) >= tan(gamma_min) * X(1,k)];
        Constraints = [Constraints, X(3,k) >= -tan(gamma_min) * X(1,k)];
        
        Constraints = [Constraints, -deg2rad(30) <= X(5,k) <= deg2rad(30)];
        Constraints = [Constraints, -deg2rad(40) <= X(6,k) <= deg2rad(40)];
        
        if k >= round(0.8 * N)
            Constraints = [Constraints, -deg2rad(5) <= X(5,k) <= deg2rad(5)];
            Constraints = [Constraints, -deg2rad(5) <= X(6,k) <= deg2rad(5)];
        end
    end
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
    
    if sol.problem ~= 0
        fprintf('Solver failed! Code: %d, Info: %s\n', sol.problem, sol.info);
        break
    end
    
    if any(any(isnan(value(X))))
        fprintf('NaN in solution, stopping.\n');
        break
    end
    
    cheat_val = value(nu_penalty);
    fprintf('Solved. Virtual Control Usage: %.4f\n', cheat_val);
    
    X_ref   = value(X);
    U_ref   = value(U);
    Gam_ref = value(Gam);

    if cheat_val < 1e-3 && iter > 2
        fprintf('Converged at iteration %d\n', iter);
        break
    end
end

fprintf('\nOptimization Complete! Final Mass: %.1f kg\n', X_ref(7,end));

t_vec = linspace(0, tf, N);

figure('Name', 'Convex Optimization Landing Trajectory');

subplot(2,2,1);

x_cone = linspace(0, X0(1), 50);
z_cone_pos =  tan(gamma_min) * x_cone;
z_cone_neg = -tan(gamma_min) * x_cone;
fill([x_cone, fliplr(x_cone)], [z_cone_pos, fliplr(z_cone_neg)], ...
     [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5); hold on;
plot(X_ref(1,:), X_ref(3,:), 'b-', 'LineWidth', 2);
plot(X_ref(1,1), X_ref(3,1), 'go', 'MarkerSize', 8, 'LineWidth', 2);
plot(X_ref(1,end), X_ref(3,end), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Downrange x (m)'); ylabel('Altitude z (m)');
title('Flight Profile (shaded = glide-slope cone)');
grid on; legend('Glide-slope zone', 'Trajectory', 'Start', 'End');

subplot(2,2,2);
plot(t_vec, X_ref(5,:) * 180/pi, 'k-', 'LineWidth', 2); hold on;
yline(5,  'r--', '+5 deg limit (terminal)');
yline(-5, 'r--', '-5 deg limit (terminal)');
yline(30,  'b--', '+30 deg limit');
yline(-30, 'b--', '-30 deg limit');
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
yline(rad2deg(p.delta_max),  'b--', 'Limit');
yline(-rad2deg(p.delta_max), 'b--', 'Limit');
xlabel('Time (s)'); ylabel('Gimbal Angle (deg)');
title('Gimbal Profile'); grid on;

function [f, A, B, C] = get_linear_dyn(X, U, Gam, p)
    vx = X(2); vz = X(4); th = X(5); w = X(6); m = X(7);
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