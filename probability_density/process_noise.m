rng(42)

global sample_rate
sample_rate = 100;

global s1
global s2
global x0
global I_sp
global g
global L
global T
global delta

I_sp = 285;
g = 9.81;
L = 41.2;

T = 5000;
delta = deg2rad(10);

x0 = [100; 100; 100; 100; deg2rad(20); deg2rad(5); 433100];

theta = x0(5);
m = x0(7);

F = get_F(theta, m);

% states to be plotted
s1 = 2;
s2 = 5;

P0 = diag([1 1 1 1 1 1 1]);

gamma = 1;

P0 = P0 * gamma; % scale P0

n = length(x0);

alpha = 0.2;
beta = 2;
k = 0;

lambda = alpha^2 * (n + k) - n;

P0_sqrt = chol(P0, 'lower');

X0 = x0;
X1 = x0 + sqrt(n + lambda) * P0_sqrt(:,1);
X2 = x0 + sqrt(n + lambda) * P0_sqrt(:,2);
X3 = x0 + sqrt(n + lambda) * P0_sqrt(:,3);
X4 = x0 + sqrt(n + lambda) * P0_sqrt(:,4);
X5 = x0 + sqrt(n + lambda) * P0_sqrt(:,5);
X6 = x0 + sqrt(n + lambda) * P0_sqrt(:,6);
X7 = x0 + sqrt(n + lambda) * P0_sqrt(:,7);
X8 = x0 - sqrt(n + lambda) * P0_sqrt(:,1);
X9 = x0 - sqrt(n + lambda) * P0_sqrt(:,2);
X10 = x0 - sqrt(n + lambda) * P0_sqrt(:,3);
X11 = x0 - sqrt(n + lambda) * P0_sqrt(:,4);
X12 = x0 - sqrt(n + lambda) * P0_sqrt(:,5);
X13 = x0 - sqrt(n + lambda) * P0_sqrt(:,6);
X14 = x0 - sqrt(n + lambda) * P0_sqrt(:,7);

wm0 = lambda / (n + lambda);
wc0 = wm0 + (1 - alpha^2 + beta);
wm1 = 1 / (2 * (n + lambda));
wc1 = wm1;
wm2 = wm1;
wc2 = wm1;
wm3 = wm1;
wc3 = wm1;
wm4 = wm1;
wc4 = wm1;
wm5 = wm1;
wc5 = wm1;
wm6 = wm1;
wc6 = wm1;
wm7 = wm1;
wc7 = wm1;
wm8 = wm1;
wc8 = wm1;
wm9 = wm1;
wc9 = wm1;
wm10 = wm1;
wc10 = wm1;
wm11 = wm1;
wc11 = wm1;
wm12 = wm1;
wc12 = wm1;
wm13 = wm1;
wc13 = wm1;
wm14 = wm1;
wc14 = wm1;

Z0 = f(X0);
Z1 = f(X1);
Z2 = f(X2);
Z3 = f(X3);
Z4 = f(X4);
Z5 = f(X5);
Z6 = f(X6);
Z7 = f(X7);
Z8 = f(X8);
Z9 = f(X9);
Z10 = f(X10);
Z11 = f(X11);
Z12 = f(X12);
Z13 = f(X13);
Z14 = f(X14);

xkp1_est = wm0 * Z0 + wm1 * Z1 + wm2 * Z2 + wm3 * Z3 + wm4 * Z4 + wm5 * Z5 ...
         + wm6 * Z6 + wm7 * Z7 + wm8 * Z8 + wm9 * Z9 + wm10 * Z10 + wm11 * Z11 ...
         + wm12 * Z12 + wm13 * Z13 + wm14 * Z14;

Pzz_SUT = wc0 * (Z0 - xkp1_est) * (Z0 - xkp1_est)' ...
        + wc1 * (Z1 - xkp1_est) * (Z1 - xkp1_est)' ...
        + wc2 * (Z2 - xkp1_est) * (Z2 - xkp1_est)' ...
        + wc3 * (Z3 - xkp1_est) * (Z3 - xkp1_est)' ...
        + wc4 * (Z4 - xkp1_est) * (Z4 - xkp1_est)' ...
        + wc5 * (Z5 - xkp1_est) * (Z5 - xkp1_est)' ...
        + wc6 * (Z6 - xkp1_est) * (Z6 - xkp1_est)' ...
        + wc7 * (Z7 - xkp1_est) * (Z7 - xkp1_est)' ...
        + wc8 * (Z8 - xkp1_est) * (Z8 - xkp1_est)' ...
        + wc9 * (Z9 - xkp1_est) * (Z9 - xkp1_est)' ...
        + wc10 * (Z10 - xkp1_est) * (Z10 - xkp1_est)' ...
        + wc11 * (Z11 - xkp1_est) * (Z11 - xkp1_est)' ...
        + wc12 * (Z12 - xkp1_est) * (Z12 - xkp1_est)' ...
        + wc13 * (Z13 - xkp1_est) * (Z13 - xkp1_est)' ...
        + wc14 * (Z14 - xkp1_est) * (Z14 - xkp1_est)';

Pzz_linear = F * P0 * F';

% Plots

gridx = linspace(95, 105, 50);
gridy = linspace(-5, 5, 50);

figure(1)
hold on
plot_ellipse([Pzz_SUT(s1,s1) Pzz_SUT(s1,s2); Pzz_SUT(s2,s1) Pzz_SUT(s2,s2)], [xkp1_est(s1); xkp1_est(s2)], 'r', 3);
plot_ellipse([Pzz_linear(s1,s1) Pzz_linear(s1,s2); Pzz_linear(s2,s1) Pzz_linear(s2,s2)], [xkp1_est(s1); xkp1_est(s2)], 'b', 3);
plot_transformed_contour(gridx, gridy, x0, P0)
legend('$3\sigma$ Unscented', '$3\sigma$ Linear', 'FontSize', 12, 'Location', 'northwest', 'Interpreter', 'latex')
title('$p(x_{k+1})$ Probability Density ($d\theta$ vs $z$)', 'FontSize', 14, 'Interpreter', 'latex')
xlabel('$z$ [m]', 'FontSize', 14, 'Interpreter', 'latex')
ylabel('$d\theta$ [rad]', 'FontSize', 14, 'Interpreter', 'latex')
grid on

% Functions

function plot_ellipse(P, m, color, sigma)
[eigvecs, eigvals] = eig(P);

theta = linspace(0, 2*pi, 1000);
circle = [cos(theta); sin(theta)];
ellipse = eigvecs * sqrt(eigvals) * (sigma * circle) + m;

plot(ellipse(1,:), ellipse(2,:), 'LineWidth', 1.5, 'Color', color);
end

function plot_transformed_contour(gridx, gridy, m, P)
global s1
global s2
global x0
global sample_rate

[gridX, gridY] = meshgrid(gridx, gridy);
gridXY_z = [gridX(:), gridY(:)];

gridXY_x = zeros(length(gridXY_z(:,1)), length(gridXY_z(1,:)));
for i = 1:length(gridXY_z(:,1))
    f_inv_state = x0;
    f_inv_state(s1) = gridXY_z(i,1);
    f_inv_state(s2) = gridXY_z(i,2);
    gridXY_x_full = f_inv(f_inv_state)';
    gridXY_x(i,:) = [gridXY_x_full(s1); gridXY_x_full(s2)];
end
m_trim(1) = m(s1);
m_trim(2) = m(s2);
px = mvnpdf(gridXY_x, m_trim, [P(s1,s1) P(s1,s2); P(s2,s1) P(s2,s2)]);
pz = zeros(length(px), 1);

for i = 1:length(px)
    z = gridXY_z(i,:);

    state0 = x0;
    state0(s1) = z(1);
    state0(s2) = z(2);

    theta = state0(5);
    m = state0(7);
    F = get_F(theta, m);
    F = expm(F*1/sample_rate);
    F_inv = inv(F);

    F_inv = [F_inv(s1,s1) F_inv(s1,s2);
             F_inv(s2,s1) F_inv(s2,s2)];

    pz(i) = px(i) * abs(det(F_inv));
end

pz = reshape(pz, length(gridy), length(gridx));
contour(gridx, gridy, pz, linspace(max(pz(:)*0), max(pz(:)), 20))
colorbar
end

function xkp1 = f(xk)
global sample_rate

stm0 = eye(7);
X0 = [xk; stm0(:)];

tf = 1/sample_rate;
num_steps = 1000;
t = linspace(0, tf, num_steps);

tol = 1E-12;
options = odeset('RelTol',tol,'AbsTol',tol);
[~, X] = ode45(@dynamics_stm, t, X0, options);

xkp1 = X(end, 1:7)';
stmf = reshape(X(end, 8:end), 7, 7);

end

function xk = f_inv(xkp1)
global sample_rate

stm0 = eye(7);
X0 = [xkp1; stm0(:)];

tf = 1/sample_rate;
num_steps = 1000;
t = linspace(tf, 0, num_steps);

tol = 1E-12;
options = odeset('RelTol',tol,'AbsTol',tol);
[~, X] = ode45(@dynamics_stm, t, X0, options);

xk = X(end, 1:7)';
stmf = reshape(X(end, 8:end), 7, 7);

end

function dstate = dynamics_stm(t, state)
global I_sp
global g
global L
global T
global delta

stm = reshape(state(8:end), 7, 7);

x = state(1);
dx = state(2);
z = state(3);
dz = state(4);
theta = state(5);
dtheta = state(6);
m = state(7);

F = get_F(theta, m);

dstm = F*stm;

dstate = [dx;
          T / m * sin(theta + delta);
          dz;
          T / m * cos(theta + delta) - g;
          dtheta;
          -6 * T / (m * L) * sin(delta);
          -T / (I_sp * g);
          dstm(:)];
end

function F = get_F(theta, m)
global T
global L
global delta

F = [0 1 0 0 0 0 0;
     0 0 0 0 (T/m)*cos(theta+delta) 0 -(T/m^2)*sin(theta+delta);
     0 0 0 1 0 0 0;
     0 0 0 0 -(T/m)*sin(theta+delta) 0 -(T/m^2)*cos(theta+delta);
     0 0 0 0 0 1 0;
     0 0 0 0 0 0 (6*T/(m^2*L))*sin(delta);
     0 0 0 0 0 0 0];
end
