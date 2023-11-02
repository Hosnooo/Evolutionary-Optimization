clear; clc; close all;
%% Parameters

runs = 10;
lb = [1;0.1;0.1]; ub = [100;1.0;0.1];
swarmsizes = randi([10 50],1,runs);
maxiterations = randi([100 300],1,runs);
C1 = 0 + 4*rand(1,runs); C2 = 4- C1;
W_init = 1 + 0.4*rand(1,runs); W_final = 0.2 + 0.2*rand(1,runs); 
global bestfvals bestsols
names = []; finalsols = []; finalfs = [];

%% Particle-Swarm

fig = figure(1);
tic
for i = 1:10

    bestfvals = []; bestsols = [];
    options = optimoptions('particleswarm','SwarmSize',swarmsizes(i),...
        'MaxIterations',maxiterations(i),'SelfAdjustmentWeight', C1(i),...
        'SocialAdjustmentWeight',C2(i),'InertiaRange',[W_final(i) W_init(i)],...
    'Display','none','OutputFcn',@Plotting);

    [x,fval,exitFlag,output] = particleswarm(@func,3,lb,ub,options);
    iterations = 1:length(bestfvals);
    plot(iterations, bestfvals,'.-', 'linewidth', 0.8)
    hold on;
    names = [names {num2str(i)}];
    finalsols = [finalsols; bestsols(end,:)];
    finalfs = [finalfs bestfvals(end)];
end
timeval = toc

grid on;
xlabel('Iteration')
ylabel('Maximum Sigma')
title('PSO Runs')
xlim([min(iterations) inf])
lgd = legend(names);
title(lgd,'Run')
% saveas(fig, 'Results.fig')
column_names = {'swarmsizes', 'maxiterations', 'C1', 'C2', 'W_init', 'W_final',...
    'K', 'T_1', 'T_2', 'Max Sigma'};
T = table(swarmsizes', maxiterations', C1', C2', W_init', W_final',...
    finalsols(:,1), finalsols(:,2), finalsols(:,3), finalfs',...
    'RowNames', names', 'VariableNames',column_names);
disp(T)
% writetable(T,'ParticleSwarmResults.xlsx',"AutoFitWidth",true,'WriteRowNames',true);

%% Plotting Packets Function

function stop = Plotting(optimValues,state)
    global bestfvals bestsols
    bestfvals = [bestfvals optimValues.bestfval];
    bestsols = [bestsols; optimValues.bestx];
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