
function [fp, tp] = roc_points(m1, cov, points1, points0, N)

thresholds = linspace(0, 0.3, 100);  % Threshold values

for i = 1:length(thresholds)   % Per ogni possibile threshold
    thresh = thresholds(i);   % scegliamo una certa threshold 

    % scelgo che la classe positivo è quella con centro [0, 0]. Quindi 

    % TP
    likelihood = mvnpdf(points1, [m1, 0], cov);
    belief = 0.5*likelihood;
    decision = (belief./(1-belief));
    predictions = (decision >= thresh);
    tp(i) = sum(predictions)/N;

    % FP
    likelihood = mvnpdf(points0, [m1, 0], cov);
    belief = 0.5*likelihood;
    decision = (belief./(1-belief));
    predictions = (decision >= thresh);
    fp(i) = sum(predictions)/N;


end

