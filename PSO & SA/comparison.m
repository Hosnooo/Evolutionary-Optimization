clear; clc; close all
%%
runs = 10;
Tpos = readtable("ParticleSwarmResults.xlsx");
Tpos = Tpos(1:runs,:);

Tsa = readtable("SimulatedAnnealingResults.xlsx");
Tsa = Tsa(1:runs,:);
y =[table2array(Tpos(:,end)) table2array(Tsa(:,end))];
plot(1:runs, y, 'linewidth', 1.2)
legend([{'PSO'} {'SA'}])
grid on;
ylim([-4 0])
title('PSO vs SA')
ylabel('Final Solution')
xlabel('iteration')
std_pos = std(y(:,1));
std_sa = std(y(:,2));