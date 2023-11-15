clear; clc; close all;
%%
%   x - input data.
%   t - target data.

% x = 2*rand(1,300);
% t = exp(-x.^2);

x = -1 + (1 - (-1))*rand(2,300);
t = sin(2*pi*x(1,:)).*cos(0.5*x(2,:)*pi);

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
net = feedforwardnet(3, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Samples Study Params
samplespercent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
samples = samplespercent*size(x,2);
Errors = [];
trs = [];
trs_perfs = [];

for i=1:length(samples)
    
    % Create a Fitting Network
    net = feedforwardnet(3, trainFcn);
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,tr] = train(net,x(:,1:samples(i)),t(:,1:samples(i)));

    % Errors
    Errors = [Errors; tr.best_perf tr.best_vperf tr.best_tperf tr.num_epochs];

    % Networks
    trs = [trs tr]; trs_perfs = [trs_perfs tr.best_perf];

    clear net
end

% Saving Results
fig = figure; plotperform(trs(trs_perfs == min(trs_perfs)));
saveas(fig, 'BestPerformance_2.png')
fig = figure; bar(samples', Errors(:,1:3), 'linewidth', 1.2); grid on;
legend({'Perfromance','Validation','Test'});
saveas(fig,'PerformanceComparison_2.png')

T = table(samples', Errors(:,1), Errors(:,2), Errors(:,3), Errors(:,4),...
    'VariableNames', {'Samples no.','Perfromance','Validation','Test','Epochs'})

writetable(T,'Results_Samples_2.csv')