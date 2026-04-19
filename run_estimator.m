rng(42)

X_0 = [1 1 1 1 1 1 1]';
P_0 = eye(7) * 1;

Q = eye(7) * 0.01^2;
R = eye(4) * 0.01^2;

const.sample_rate = 10;
const.g = 9.81;
const.I_sp = 285;
const.L = 41.2;

tf = 10;
t = linspace(0, tf, tf * const.sample_rate);

% Generate a truth state history with some dummy control inputs
U.T = 1;
U.delta = 1;
options = odeset('RelTol', 1E-12,'AbsTol', 1E-12);
[t, X_truth] = ode45(@dummy_dynamics, t, X_0, options, U, const);

% Start with perfect state and covariancle knowledge
X_est = X_0;
P_est = P_0;

X_est_hist = zeros(length(t), 7);
P_est_hist = zeros(7, 7, length(t));
for i = 1:length(t)
    % Dummy sinusoidal control inputs.
    U_dummy.T = abs(U.T * sin(t(i)));
    U_dummy.delta = U.delta * sin(t(i));

    % Get measurements
    z_truth = h(X_truth(i,:)');
    z = z_truth + 0.01 * randn(4, 1);

    % TODO Replace z(3) with angle measurement from attitude EKF
    % z(3) = ...

    %[X_est, P_est] = position_EKF(X_est, P_est, z, Q, R, U_dummy, const);
    [X_est, P_est] = position_UKF(X_est, P_est, z, Q, R, U_dummy, const);

    X_est_hist(i,:) = X_est;
    P_est_hist(:,:,i) = P_est;
end

figure(1)
for i = 1:7
    subplot(7, 1, i)
    hold on
    plot(t, X_est_hist(:,i) - X_truth(:,i))
    plot(t, 3 * sqrt(squeeze(P_est_hist(i, i, :))))
    plot(t, -3 * sqrt(squeeze(P_est_hist(i, i, :))))
end
    
figure(2)
for i = 1:7
    subplot(7, 1, i)
    hold on
    plot(t, X_truth(:,i))
    plot(t, X_est_hist(:,i))
end

function dX = dummy_dynamics(t, X, U, const)
% Dummy dynamics with sinusoidal control inputs
I_sp = const.I_sp;
g = const.g;
L = const.L;

T = abs(U.T * sin(t));
delta = U.delta * sin(t);

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