




%% Naive kernel
% L'obiettivo è: per ogni x, prendere i vicini entro una distanza h
% r_n sarà la media delle label delle x che cadono nell'intervallo 

% TODO 
% a = 2*pi;
% step = 0.005

function [mse] = monteCarlo(Nmc, step, a)

for round = [1:1:Nmc]
    [x, y, ytrue] = dataGeneration(step, a);
    % calcolo di rn 
    
end
mse = mean((y-rn).^2);




%% Nearest n
%% Plottare le funzioni di regressione (?) 