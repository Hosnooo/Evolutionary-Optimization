clc; clear; close all;
%% Fuzzy System Creation

% Create fuzzy inference system
fis = mamfis("Name","FIS_ControlSystem");

% Add input and output variables to the system
fis = addInput(fis, [-10 10], "Name", "x1");
fis = addInput(fis, [-10 10], "Name", "x2");
fis = addOutput(fis, [-24 24], "Name", "u");

% Add membership functions of inputs and outputs
fis = addMF(fis,"x1","linzmf",[-2 0],"Name","N");
fis = addMF(fis,"x1","linsmf",[0 2],"Name","P");
fis = addMF(fis,"x1","trimf",[-2 0 2],"Name","Z");

fis = addMF(fis,"x2","linzmf",[-5 0],"Name","N");
fis = addMF(fis,"x2","linsmf",[0 5],"Name","P");
fis = addMF(fis,"x2","trimf",[-5 0 5],"Name","Z");

fis = addMF(fis,"u","linzmf",[-16 -8],"Name","NB");
fis = addMF(fis,"u","trimf",[-16 -8 0],"Name","N");
fis = addMF(fis,"u","trimf",[-8 0 8],"Name","Z");
fis = addMF(fis,"u","trimf",[0 8 16],"Name","P");
fis = addMF(fis,"u","linsmf",[8 16],"Name","PB");

% Add rules to the system
rules = [...
    "x1==P & x2==P => u=PB",...
    "x1==P & x2==Z => u=P",...
    "x1==P & x2==N => u=Z",...
    "x1==Z & x2==P => u=P",...
    "x1==Z & x2==Z => u=Z",...
    "x1==Z & x2==N => u=N",...
    "x1==N & x2==P => u=Z",...
    "x1==N & x2==Z => u=N",...
    "x1==N & x2==N => u=NB"
    ];

fis = addRule(fis,rules);

showrule(fis)
plotfis(fis)

% writeFIS(fis);
% save('controller_data.mat','fis');

%% Input-Output Mapping

% Simulation Parameters and initial conditions
x1_0 = 1; x2_0 = -4; cycles = 4;

% Packets
input_packet = [x1_0 x2_0]; output_packet = [];
% opt = evalfisOptions('NumSamplePoints',1e3);

x1_val = x1_0; x2_val = x2_0;
for i = 1:cycles
    [u_val, fuzzifiedIn, ruleOut, aggregatedOut, ruleFiring] = ...
        evalfis(fis, [x1_val x2_val]);
    % x = linspace(-24,24,101);
    % u_val = defuzz(x,aggregatedOut,'centroid');
    x1_old = x1_val;
    x2_old = x2_val;
    x1_val = x1_old + x2_old;
    x2_val = x1_old + x2_old - u_val;
    input_packet = [input_packet; x1_val x2_val];
    output_packet = [output_packet; u_val];
end
input_packet(end,:) = [];
%% Plots
figure;
plotmf(fis,"input",1);
title("X1 Membership");
xlabel("X1");
ylabel("Membership Value");
grid on;
legend("N","Z","P");
legend("Location", "best");
limitIncreaseFactor = 0.05;
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

figure;
plotmf(fis,"input",2);
title("X2 Membership");
xlabel("X2");
ylabel("Membership Value");
grid on;
legend("N","Z","P");
legend("Location", "east");
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

figure;
plotmf(fis,"output",1);
title("u Membership");
xlabel("u");
ylabel("Membership Value");
grid on;
legend("NB","N","Z","P","PB");
legend("Location", "east");
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);

figure;
plot(1:cycles,input_packet(:,1),"linewidth",0.8)
hold on;
plot(1:cycles,input_packet(:,2),"linewidth",0.8)
plot(1:cycles,output_packet,'k-.',"linewidth",0.8)
grid on;
title('Fuzzy Simulation')
xlabel('Cycle')
ylabel('unit input/output')
legend('X1','X2','u')
newLimits = edit_limits(axis, limitIncreaseFactor);
axis(newLimits);