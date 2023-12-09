clc; clear; close all;
load('controller_data.mat')

%% Simulation Parameters and initial conditions
x1_0 = 1; x2_0 = -4; cycles = 4;
x = linspace(-24,24,101);

% Packets
output_packet = [];

x1_val_c = x1_0; x2_val_c = x2_0;
x1_val_b = x1_0; x2_val_b = x2_0;
x1_val_l = x1_0; x2_val_l = x2_0;
x1_val_s = x1_0; x2_val_s = x2_0;
input_packet = [x1_val_c, x2_val_c x1_val_b x2_val_b, x1_val_s x2_val_s];

for i = 1:cycles
    [~, ~, ~, aggregatedOut, ~] = ...
        evalfis(fis, [x1_val_c x2_val_c]);
    u_val_c = defuzz(x,aggregatedOut,'centroid');
    x1_old_c = x1_val_c;
    x2_old_c = x2_val_c;
    x1_val_c = x1_old_c + x2_old_c;
    x2_val_c = x1_old_c + x2_old_c - u_val_c;
    
    [~, ~, ~, aggregatedOut, ~] = ...
        evalfis(fis, [x1_val_b x2_val_b]);
    u_val_b = defuzz(x,aggregatedOut,'bisector');
    x1_old_b = x1_val_b;
    x2_old_b = x2_val_b;
    x1_val_b = x1_old_b + x2_old_b;
    x2_val_b = x1_old_c + x2_old_b - u_val_b;

    % [~, ~, ~, aggregatedOut, ~] = ...
    %     evalfis(fis, [x1_val_l x2_val_l]);
    % u_val_l = defuzz(x,aggregatedOut,'lom');
    % x1_old_l = x1_val_l;
    % x2_old_l = x2_val_l;
    % x1_val_l = x1_old_l + x2_old_l;
    % x2_val_l = x1_old_l + x2_old_l - u_val_l;

    [~, ~, ~, aggregatedOut, ~] = ...
        evalfis(fis, [x1_val_s x2_val_s]);
    u_val_s = defuzz(x,aggregatedOut,'som');
    x1_old_s = x1_val_s;
    x2_old_s = x2_val_s;
    x1_val_s = x1_old_s + x2_old_s;
    x2_val_s = x1_old_s + x2_old_s - u_val_s;
    input_packet = [input_packet; x1_val_c x2_val_c x1_val_b x2_val_b x1_val_s x2_val_s];
    output_packet = [output_packet; u_val_c u_val_b u_val_s];
end
% input_packet(:,end-5:end) = [];
writematrix(output_packet, 'outputs.xlsx')
writematrix(input_packet, 'inputs.xlsx')

%% Plots
figure;
pp = plot(1:cycles,output_packet);
grid on;
hold on;
plot(1:cycles,[-2 -9.6 0 8.9667],'-.')
title('Fuzzy Simulation')
xlabel('Cycle')
ylabel('Unit Output')
legend('Centroid','Bisector','Som', 'Weighted Mean')

limitIncreaseFactor = 0.1;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

figure;
subplot(2,2,1)
plot(1:cycles,input_packet(1:4,1:2));
grid on;
title('Centroid Inputs')
xlabel('Cycle')
ylabel('Unit Input')
legend('X1','X2')

limitIncreaseFactor = 0.1;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

subplot(2,2,2)
plot(1:cycles,input_packet(1:4,3:4));
grid on;
title('Birsctor Inputs')
xlabel('Cycle')
ylabel('Unit Input')
legend('X1','X2')

limitIncreaseFactor = 0.1;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

subplot(2,2,3)
plot(1:cycles,input_packet(1:4,5:6));
grid on;
title('Som Inputs')
xlabel('Cycle')
ylabel('Unit Input')
legend('X1','X2')

limitIncreaseFactor = 0.1;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

subplot(2,2,4)
plot(1:cycles,[1 -4; -3 -1; -4 5.6; 1.6 1.6]);
grid on;
title('Weighted Means Inputs')
xlabel('Cycle')
ylabel('Unit Input')
legend('X1','X2')

limitIncreaseFactor = 0.1;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);