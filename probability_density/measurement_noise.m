rng(42)
global theta
theta = deg2rad(30);

x0 = [0, 100]';
P0 = diag([0.1^2 0.001^2]);

gamma = 1000;

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
X3 = x0 - sqrt(n + lambda) * P0_sqrt(:,1);
X4 = x0 - sqrt(n + lambda) * P0_sqrt(:,2);

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

Z0 = h(X0);
Z1 = h(X1);
Z2 = h(X2);
Z3 = h(X3);
Z4 = h(X4);

z_est = wm0 * Z0 + wm1 * Z1 + wm2 * Z2 + wm3 * Z3 + wm4 * Z4;
mz = z_est;

Pzz_SUT = wc0 * (Z0 - z_est) * (Z0 - z_est)' ...
        + wc1 * (Z1 - z_est) * (Z1 - z_est)' ...
        + wc2 * (Z2 - z_est) * (Z2 - z_est)' ...
        + wc3 * (Z3 - z_est) * (Z3 - z_est)' ...
        + wc4 * (Z4 - z_est) * (Z4 - z_est)';

H = [x0(1)/sqrt(x0(1)^2+x0(2)^2) x0(2)/sqrt(x0(1)^2+x0(2)^2);
     -x0(2)/(x0(1)^2+x0(2)^2) x0(1)/(x0(1)^2+x0(2)^2)];

Pzz_linear = H * P0 * H';

% Plots

gridx = linspace(99.8, 100.3, 100);
gridy = linspace(-0.65, -0.4, 100);

figure(1)
hold on
plot_ellipse(Pzz_SUT, mz, 'r', 3);
plot_ellipse(Pzz_linear, mz, 'b', 3);
plot_transformed_contour(gridx, gridy, x0, P0)
legend('$3\sigma$ Unscented', '$3\sigma$ Linear', 'FontSize', 12, 'Location', 'northwest', 'Interpreter', 'latex')
title('$p(z)$ Probability Density ($\alpha_m$ vs $d_m$)', 'FontSize', 14, 'Interpreter', 'latex')
xlabel('$d_m$ [m]', 'FontSize', 14, 'Interpreter', 'latex')
ylabel('$\alpha_m$ [rad]', 'FontSize', 14, 'Interpreter', 'latex')
grid on

% Functions

function z = h(x)
global theta
z(1) = sqrt(x(1)^2 + x(2)^2);
z(2) = -atan2(x(1), x(2)) - theta;
z = z';

end

function x = h_inv(z)
global theta
x(1) = -z(1) * sin(theta + z(2));
x(2) = z(1) * cos(theta + z(2));
x = x';
end

function plot_ellipse(P, m, color, sigma)
%P = abs(P);
[eigvecs, eigvals] = eig(P);

theta = linspace(0, 2*pi, 1000);
circle = [cos(theta); sin(theta)];
ellipse = eigvecs * sqrt(eigvals) * (sigma * circle) + m;

plot(ellipse(1,:), ellipse(2,:), 'LineWidth', 1.5, 'Color', color);
end

function plot_transformed_contour(gridx, gridy, m, P)
global theta

[gridX, gridY] = meshgrid(gridx, gridy);
gridXY_z = [gridX(:), gridY(:)];

gridXY_x = zeros(length(gridXY_z(:,1)), length(gridXY_z(1,:)));
for i = 1:length(gridXY_z(:,1))
    gridXY_x(i,:) = h_inv(gridXY_z(i,:))';
end

px = mvnpdf(gridXY_x, m', P);
pz = zeros(length(px), 1);

for i = 1:length(px)
    z = gridXY_z(i,:);

    H_inv = [-sin(theta+z(2)) -z(1)*cos(theta+z(2));
             cos(theta+z(2))  -z(1)*sin(theta+z(2))];
    pz(i) = px(i) * abs(det(H_inv));
end

pz = reshape(pz, length(gridy), length(gridx));
contour(gridx, gridy, pz, linspace(min(pz(:)), max(pz(:)), 20))
colorbar
end




