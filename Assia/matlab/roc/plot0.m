
clear; close all; clc;

%% Generazione dei dati 

N = 50;
m0 = 5;


%% Plot della roc 

% [fp, tp] = roc_points(0, cov, points1, points0, N);
% 
% figure (2);
% plot(fp, tp, color='red');

figure()
NSigma = [0.1, 0.5, 1, 3, 5, 7, 10, 15, 18, 20];
colors = jet(length(NSigma));


for i=1:length(NSigma)

    [points1,points0] = generation(m0, NSigma(i), N);
    NSigma(i)
    cov = diag(NSigma(i)*ones(1, 2));
    [fp, tp] = roc_points(0, cov, points1, points0, N);
    plot(fp, tp, color=colors(i, :));
    legend_str{i} = sprintf(['var =',num2str(NSigma(i))]);
    hold on 
end

hold on 
plot(fp, fp, 'black', 'LineWidth', 1.5);



legend(legend_str, 'Location', 'southeast');
xlabel('Falsi positivi');
ylabel('Veri positivi');
title('ROC al variare di \sigma^2');





%% Likelihood e decisione 

% center = [5, 0];          % Center of the distribution
% covarianceMatrix = eye(2); % Covariance matrix (identity matrix for simplicity)
% point = [4, 1];            % Point for which to calculate likelihood
% 
% % Calculate the likelihood using the normal distribution PDF
% likelihood = mvnpdf(point, center, covarianceMatrix);
% belief = 0.5*likelihood;
% decision = (belief/(1+belief))
% % prob = 1/(1+exp(-decision))
% 
% 
% point = [3, 0];
% likelihood = mvnpdf(point, center, covarianceMatrix);
% belief = 0.5*likelihood;
% decision = (belief/(1+belief))
% % prob = 1/(1+exp(-decision))
% 
% 
% likelihood = mvnpdf(points1, [0, 0], covarianceMatrix);
% belief = 0.5*likelihood;
% decision = log10(belief./(1-belief));
% thresh = 0.001; 
% predictions = (decision <= thresh);
% score = sum(predictions)/N



