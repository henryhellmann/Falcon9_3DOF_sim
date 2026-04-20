function [X_est, P_est] = position_UKF(X_0, P_0, z, Q, R, U, const)
[X_pred, P_pred] = UKF_propagation(X_0, P_0, Q, U, const);
[X_est, P_est] = UKF_update(X_pred, P_pred, R, z);
end

function [X_f, P_f] = UKF_propagation(X_0, P_0, Q, U, const)
tf = 1 / const.sample_rate;
num_steps = 100;
t = linspace(0, tf, num_steps);

% tol = 1E-12;
% options = odeset('RelTol', tol,'AbsTol', tol);
% [~, X] = ode45(@dynamics, t, X_0, options, U, const);

% X_f = X(end,:)';
[X_f, P_f] = propagation_unscented_transform(X_0, P_0, Q, U, const);
end

function [X_est, P_est] = UKF_update(X_pred, P_pred, R, z)
[z_pred, Pz, Pxz] = update_unscented_transform(X_pred, P_pred, R);

K = Pxz/Pz;

X_est = X_pred + K*(z - z_pred);
P_est = P_pred - K*Pz*K';
P_est = (P_est + P_est')/2; % keep symmetry
end

function [X_f, P_f] = propagation_unscented_transform(X_0, P_0, Q, U, const)
n = length(X_0);
alpha = 1e-3;
beta = 2;
k = 3-n;
lambda = alpha^2 * (n + k) - n;

P_0 = (P_0 + P_0')/2;
P_0_sqrt = chol(P_0, 'lower');

X0_0  = X_0;
X1_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,1);
X2_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,2);
X3_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,3);
X4_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,4);
X5_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,5);
X6_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,6);
X7_0  = X_0 + sqrt(n + lambda) * P_0_sqrt(:,7);
X8_0  = X_0 - sqrt(n + lambda) * P_0_sqrt(:,1);
X9_0  = X_0 - sqrt(n + lambda) * P_0_sqrt(:,2);
X10_0  = X_0 - sqrt(n + lambda) * P_0_sqrt(:,3);
X11_0  = X_0 - sqrt(n + lambda) * P_0_sqrt(:,4);
X12_0 = X_0 - sqrt(n + lambda) * P_0_sqrt(:,5);
X13_0 = X_0 - sqrt(n + lambda) * P_0_sqrt(:,6);
X14_0 = X_0 - sqrt(n + lambda) * P_0_sqrt(:,7);

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

tf = 1 / const.sample_rate;
num_steps = 100;
t = linspace(0, tf, num_steps);
options = odeset('RelTol', 1E-12,'AbsTol', 1E-12);

[~, X0_f] = ode45(@dynamics, t, X0_0, options, U, const);
X0_f = X0_f(end,:)';
[~, X1_f] = ode45(@dynamics, t, X1_0, options, U, const);
X1_f = X1_f(end,:)';
[~, X2_f] = ode45(@dynamics, t, X2_0, options, U, const);
X2_f = X2_f(end,:)';
[~, X3_f] = ode45(@dynamics, t, X3_0, options, U, const);
X3_f = X3_f(end,:)';
[~, X4_f] = ode45(@dynamics, t, X4_0, options, U, const);
X4_f = X4_f(end,:)';
[~, X5_f] = ode45(@dynamics, t, X5_0, options, U, const);
X5_f = X5_f(end,:)';
[~, X6_f] = ode45(@dynamics, t, X6_0, options, U, const);
X6_f = X6_f(end,:)';
[~, X7_f] = ode45(@dynamics, t, X7_0, options, U, const);
X7_f = X7_f(end,:)';
[~, X8_f] = ode45(@dynamics, t, X8_0, options, U, const);
X8_f = X8_f(end,:)';
[~, X9_f] = ode45(@dynamics, t, X9_0, options, U, const);
X9_f = X9_f(end,:)';
[~, X10_f] = ode45(@dynamics, t, X10_0, options, U, const);
X10_f = X10_f(end,:)';
[~, X11_f] = ode45(@dynamics, t, X11_0, options, U, const);
X11_f = X11_f(end,:)';
[~, X12_f] = ode45(@dynamics, t, X12_0, options, U, const);
X12_f = X12_f(end,:)';
[~, X13_f] = ode45(@dynamics, t, X13_0, options, U, const);
X13_f = X13_f(end,:)';
[~, X14_f] = ode45(@dynamics, t, X14_0, options, U, const);
X14_f = X14_f(end,:)';

% X_f = wm0*X0_f + wm1*X1_f + wm2*X2_f + wm3*X3_f + wm4*X4_f + wm5*X5_f + wm6*X6_f + wm7*X7_f + wm8*X8_f + wm9*X9_f + wm10*X10_f + wm11*X11_f + wm12*X12_f + wm13*X13_f + wm14*X14_f;
% x_f = wm0*X0_f + wm1*X1_f + wm2*X2_f + wm3*X3_f + wm4*X4_f + wm5*X5_f + wm6*X6_f + wm7*X7_f + wm8*X8_f + wm9*X9_f + wm10*X10_f + wm11*X11_f + wm12*X12_f + wm13*X13_f + wm14*X14_f;
% x_f = X0_f;

X_f = wm0*X0_f + wm1*X1_f + wm2*X2_f + wm3*X3_f + wm4*X4_f + wm5*X5_f + wm6*X6_f + wm7*X7_f + wm8*X8_f + wm9*X9_f + wm10*X10_f + wm11*X11_f + wm12*X12_f + wm13*X13_f + wm14*X14_f;

% Calculate Covariance using the capitalized X_f
P_f = wc0*(X0_f-X_f)*(X0_f-X_f)' ...
    + wc1*(X1_f-X_f)*(X1_f-X_f)' ...
    + wc2*(X2_f-X_f)*(X2_f-X_f)' ...
    + wc3*(X3_f-X_f)*(X3_f-X_f)' ...
    + wc4*(X4_f-X_f)*(X4_f-X_f)' ...
    + wc5*(X5_f-X_f)*(X5_f-X_f)' ...
    + wc6*(X6_f-X_f)*(X6_f-X_f)' ...
    + wc7*(X7_f-X_f)*(X7_f-X_f)' ...
    + wc8*(X8_f-X_f)*(X8_f-X_f)' ...
    + wc9*(X9_f-X_f)*(X9_f-X_f)' ...
    + wc10*(X10_f-X_f)*(X10_f-X_f)' ...
    + wc11*(X11_f-X_f)*(X11_f-X_f)' ...
    + wc12*(X12_f-X_f)*(X12_f-X_f)' ...
    + wc13*(X13_f-X_f)*(X13_f-X_f)' ...
    + wc14*(X14_f-X_f)*(X14_f-X_f)';

G = eye(7);
P_f = P_f + G*Q*G';
end

function [z_est, Pz, Pxz] = update_unscented_transform(X_pred, P_pred, R)
n = length(X_pred);
alpha = 1e-3;
beta = 2;
k = 3-n;
lambda = alpha^2 * (n + k) - n;

P_pred = (P_pred + P_pred')/2;
P_pred_sqrt = chol(P_pred, 'lower');

X0  = X_pred;
X1  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,1);
X2  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,2);
X3  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,3);
X4  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,4);
X5  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,5);
X6  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,6);
X7  = X_pred + sqrt(n + lambda) * P_pred_sqrt(:,7);
X8  = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,1);
X9  = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,2);
X10  = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,3);
X11  = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,4);
X12 = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,5);
X13 = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,6);
X14 = X_pred - sqrt(n + lambda) * P_pred_sqrt(:,7);

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

Z0 = h(X0);
Z1 = h(X1);
Z2 = h(X2);
Z3 = h(X3);
Z4 = h(X4);
Z5 = h(X5);
Z6 = h(X6);
Z7 = h(X7);
Z8 = h(X8);
Z9 = h(X9);
Z10 = h(X10);
Z11 = h(X11);
Z12 = h(X12);
Z13 = h(X13);
Z14 = h(X14);

z_est = wm0*Z0 + wm1*Z1 + wm2*Z2 + wm3*Z3 + wm4*Z4 + wm5*Z5 + wm6*Z6 + wm7*Z7 + wm8*Z8 + wm9*Z9 + wm10*Z10 + wm11*Z11 + wm12*Z12 + wm13*Z13 + wm14*Z14;

Pz = wc0*(Z0-z_est)*(Z0-z_est)' ...
   + wc1*(Z1-z_est)*(Z1-z_est)' ...
   + wc2*(Z2-z_est)*(Z2-z_est)' ...
   + wc3*(Z3-z_est)*(Z3-z_est)' ...
   + wc4*(Z4-z_est)*(Z4-z_est)' ...
   + wc5*(Z5-z_est)*(Z5-z_est)' ...
   + wc6*(Z6-z_est)*(Z6-z_est)' ...
   + wc7*(Z7-z_est)*(Z7-z_est)' ...
   + wc8*(Z8-z_est)*(Z8-z_est)' ...
   + wc9*(Z9-z_est)*(Z9-z_est)' ...
   + wc10*(Z10-z_est)*(Z10-z_est)' ...
   + wc11*(Z11-z_est)*(Z11-z_est)' ...
   + wc12*(Z12-z_est)*(Z12-z_est)' ...
   + wc13*(Z13-z_est)*(Z13-z_est)' ...
   + wc14*(Z14-z_est)*(Z14-z_est)';
Pz = Pz + R;

Pxz = wc0*(X0-X_pred)*(Z0-z_est)' ...
    + wc1*(X1-X_pred)*(Z1-z_est)' ...
    + wc2*(X2-X_pred)*(Z2-z_est)' ...
    + wc3*(X3-X_pred)*(Z3-z_est)' ...
    + wc4*(X4-X_pred)*(Z4-z_est)' ...
    + wc5*(X5-X_pred)*(Z5-z_est)' ...
    + wc6*(X6-X_pred)*(Z6-z_est)' ...
    + wc7*(X7-X_pred)*(Z7-z_est)' ...
    + wc8*(X8-X_pred)*(Z8-z_est)' ...
    + wc9*(X9-X_pred)*(Z9-z_est)' ...
    + wc10*(X10-X_pred)*(Z10-z_est)' ...
    + wc11*(X11-X_pred)*(Z11-z_est)' ...
    + wc12*(X12-X_pred)*(Z12-z_est)' ...
    + wc13*(X13-X_pred)*(Z13-z_est)' ...
    + wc14*(X14-X_pred)*(Z14-z_est)';
end

function dX = dynamics(t, X, U, const)
I_sp = const.I_sp;
g = const.g;
L = const.L;

T = U.T;
delta = U.delta;

x = X(1);
dx = X(2);
z = X(3);
dz = X(4);
theta = X(5);
dtheta = X(6);
m = X(7);

dX = [dx;
      T / m * sin(theta + delta);
      dz;
      T / m * cos(theta + delta) - g;
      dtheta;
      -6 * T / (m * L) * sin(delta);
      -T / (I_sp * g)];
end

function F = get_F(X, U, const)
L = const.L;

T = U.T;
delta = U.delta;

theta = X(5);
m = X(7);

F = [0 1 0 0 0 0 0;
     0 0 0 0 (T/m)*cos(theta+delta) 0 -(T/m^2)*sin(theta+delta);
     0 0 0 1 0 0 0;
     0 0 0 0 -(T/m)*sin(theta+delta) 0 -(T/m^2)*cos(theta+delta);
     0 0 0 0 0 1 0;
     0 0 0 0 0 0 (6*T/(m^2*L))*sin(delta);
     0 0 0 0 0 0 0];
end

function H = get_H(X)
x = X(1);
z = X(3);

H = [x/sqrt(x^2+z^2) 0 z/sqrt(x^2+z^2) 0  0 0 0;
     -z/(x^2+z^2)    0 x/(x^2+z^2)     0 -1 0 0;
     0               0 0               0  1 0 0;
     0               0 0               0  0 0 1];
end

function z_meas = h(X)
x = X(1);
z = X(3);
theta = X(5);
m = X(7);

z_meas = [sqrt(x^2 + z^2);
          -atan2(x, z) - theta;
          theta;
          m];
end

function dXP = dynamics_covariance(t, XP, Q, U, const)
X = XP(1:7);
P = reshape(XP(8:end), 7, 7);

dX = dynamics(t, X, U, const);

F = get_F(X, U, const);
G = eye(7);
dP = F*P + P*F' + G*Q*G';

dXP = [dX; dP(:)];
end