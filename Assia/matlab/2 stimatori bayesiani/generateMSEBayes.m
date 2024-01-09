% Genera i campioni e calcola l'MSE. Versione per stimatore Bayesiano

% I parametri sono:
% N: numero di campioni 
% sigma: *varianza* dei campioni (non deviazione standard) 
% t: il valore vero del parametro da stimare 
% nMC: numero di Monte Carlo runs

% L'idea è che ad ogni simulazione Monte Carlo viene estratta una nuova y e
% viene generato un nuovo vetore di x su cui poi verrà caloclato lo
% stimatore e l'mse

function[MSE_ML, MSE_BAY, mse] = generateMSEBayes(N, nMC, muy, sigmay, sigmaw)


for i=1:nMC
    % Generazione dei dati 
    y(i) = normrnd(muy, sqrt(sigmay));  % estrazione di un campione Y
    w = normrnd(0, sqrt(sigmaw), 1, N);  % generazione di N campioni di rumore 
    x = y(i)+w;   % Vettore delle x 
    
    % Calcolo degli stimatori 
    t_ml(i) = mean(x);  % Lo stimatore ML non è altro che la media dei campioni 
    t_bay(i) = (sigmay/(N*sigmay + sigmaw))*sum(x);  % Stimatore bayesiano/media a posteriori 

end


% Calcolo degli errori 
MSE_ML = mean((y-t_ml).^2);
MSE_BAY = mean((y-t_bay).^2);
mse = (sigmay*(sigmaw/N))/(sigmay + (sigmaw/N));







