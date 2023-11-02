clear; clc; close all;
%% Parameters

runs = 10;
lb = [1;0.1;0.1]; ub = [100;1.0;0.1];
maxiterations = randi([100 300],1,runs);
initial_temperature = 100 + 200*rand(1,runs);
reannealing_interval = randi([100 300],1,runs);
global bestfvals bestsols temps
names = []; finalsols = []; finalfs = []; tempers = [];

%% Simulated-Annealing

fig = figure(1);
X_0 = [];
tic
for i = 1:10

    bestfvals = []; bestsols = []; temps = [];
    X_0 = lb + (ub - lb).*rand(length(lb),1);
    X_0 = X_0';
    options = optimoptions('simulannealbnd','MaxIterations',maxiterations(i),...
        'InitialTemperature', initial_temperature(i),...
        'ReannealInterval', reannealing_interval(i), ...
    'Display','none','OutputFcn',@Plotting);

    [x,fval,exitFlag,output] = simulannealbnd(@func,X_0,lb,ub,options);
    iterations = 1:length(bestfvals);
    plot(iterations, bestfvals,'.-', 'linewidth', 0.8)
    hold on;
    names = [names {num2str(i)}];
    finalsols = [finalsols; bestsols(end,:)];
    finalfs = [finalfs bestfvals(end)];
    tempers = [tempers; {temps}];

end
timerval = toc

grid on;
xlabel('Iteration')
ylabel('Maximum Sigma')
title('SA Runs')
xlim([min(iterations) inf])
lgd = legend(names);
title(lgd,'Run')
% saveas(fig, 'Results_SA.fig')

column_names = {'MaxIterations','InitialTemperature','ReannealInterval',...
    'K', 'T_1', 'T_2', 'Max Sigma'};
T = table(maxiterations', initial_temperature', reannealing_interval',...
    finalsols(:,1), finalsols(:,2), finalsols(:,3), finalfs',...
    'RowNames', names', 'VariableNames',column_names);
disp(T)

% writetable(T,'SimulatedAnnealingResults.xlsx',"AutoFitWidth",true,'WriteRowNames',true);

%% Plotting Packets Function

function [stop,options,optchanged] = Plotting(options,optimValues,flag)
    global bestfvals bestsols temps
    bestfvals = [bestfvals optimValues.bestfval];
    bestsols = [bestsols; optimValues.bestx];
    temps = [temps; optimValues.temperature];
    optchanged = [];
    stop = false;
end

%% Objective Function
function cost = func(gains)

A = [ [0 377 0 0];
    [-0.0587 0 -0.1303 0];
    [-0.0899 0 -0.1956 0.1289];
    [95.605 0 -816.0862 -20]
    ];
B = [0 0 0 0; 0 0 0 1000]';
KCL = [[-0.0587 0 -0.1303 0];
    [-0.0587*gains(1)*gains(2)/gains(3) 0 -0.1303*gains(1)*gains(2)/gains(3) 0]
    ];
BCL = [-0.333 0; gains(1)/gains(3)*(1-gains(2)/3) -1/gains(3)];

Ac = [A B; KCL BCL];

closedloopeigenvalues = eig(Ac);
real_idxs = find(imag(closedloopeigenvalues) ~= 0);
sigmas = real(closedloopeigenvalues);
osci_modes_sigmas = sigmas(real_idxs);

pos_eig = find(real(closedloopeigenvalues) >= 0);

if (~isempty(pos_eig))
    cost = inf;
else
    cost = max(osci_modes_sigmas);
end

end