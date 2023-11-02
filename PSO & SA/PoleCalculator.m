clc; clear ; close all;
%% Open Loop Eigen-Values
A = [ [0 377 0 0];
    [-0.0587 0 -0.1303 0];
    [-0.0899 0 -0.1956 0.1289];
    [95.605 0 -816.0862 -20]
    ];
%% Closed Loop Eigen-Values
k = 20.76408570; t1 = 0.26042731; t2 = 0.1;
B = [0 0 0 0; 0 0 0 1000]';
KCL = [[-0.0587 0 -0.1303 0];
    [-0.0587*k*t1/t2 0 -0.1303*k*t1/t2 0]
    ];
BCL = [-0.333 0; k/t2*(1-t1/3) -1/t2];

Ac_i = [A B; KCL BCL];
poles = eig(Ac_i);
disp('Closed loop Ac: ')
disp(Ac_i)

% Plot the poles in the complex plane
figure;
plot(real(poles), imag(poles), 'x', 'LineWidth', 1.2);  % Plot poles as 'x' marks
grid on;
xlabel('Real Axis');
ylabel('Imaginary Axis');
xline(0); yline(0)
xlim([-inf 2])
title('Pole Plot');