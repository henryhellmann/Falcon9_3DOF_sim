function [X_est, P_est] = position_EKF(X_0, P_0, z, Q, R, U, const)
[X_pred, P_pred] = EKF_propagation(X_0, P_0, Q, U, const);
[X_est, P_est] = EKF_update(X_pred, P_pred, R, z);
end

function [X_f, P_f] = EKF_propagation(X_0, P_0, Q, U, const)
XP_0 = [X_0; P_0(:)];

tf = 1 / const.sample_rate;
num_steps = 100;
t = linspace(0, tf, num_steps);

tol = 1E-12;
options = odeset('RelTol', tol,'AbsTol', tol);
[~, X] = ode45(@dynamics_covariance, t, XP_0, options, Q, U, const);

X_f = X(end, 1:7)';
P_f = reshape(X(end,8:end), 7, 7);
end

function [X_est, P_est] = EKF_update(X_pred, P_pred, R, z)
H = get_H(X_pred);

W = H*P_pred*H' + R;
C = P_pred*H';
K = C/W;

% z_pred = H*x_pred;
z_pred = h(X_pred);
X_est = X_pred + K*(z - z_pred);

% P_est = P_pred - C*K' - K*C' + K*W*K';
P_est = (eye(7) - K*H) * P_pred * (eye(7) - K*H)' + K*R*K';
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