clear; clc; close all;
%%
%   x - input data.
%   t - target data.

x = 2*rand(1,240);
t = exp(-x.^2);

% x = -1 + (1 - (-1))*rand(2,240);
% t = sin(2*pi*x(1,:)).*cos(0.5*x(2,:)*pi);

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.


% Nodes Study Params
hiddenLayerSizes = [2 3 4 5 6 7 8 9 10];
Errors = [];
trs = [];
trs_perfs = [];

for i=1:length(hiddenLayerSizes)

    % Create a Fitting Network
    net = feedforwardnet(hiddenLayerSizes(i), trainFcn);

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    % Train the Network
    [net,tr] = train(net,x,t);

    % Errors
    Errors = [Errors; tr.best_perf tr.best_vperf tr.best_tperf tr.num_epochs];

    % Networks
    trs = [trs tr]; trs_perfs = [trs_perfs tr.best_perf];

    clear net
end

% Saving Results
fig = figure; plotperform(trs(trs_perfs == min(trs_perfs)));
saveas(fig, 'BestPerformance_n1.png')
fig = figure; bar(hiddenLayerSizes', Errors(:,1:3), 'linewidth', 1.2); grid on;
legend({'Perfromance','Validation','Test'});
saveas(fig,'PerformanceComparison_n1.png')

T = table(hiddenLayerSizes', Errors(:,1), Errors(:,2), Errors(:,3), Errors(:,4),...
    'VariableNames', {'Nodes no.','Perfromance','Validation','Test','Epochs'})

writetable(T,'Results_Samples_n1.csv')