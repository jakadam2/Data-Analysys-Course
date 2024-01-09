%N: data size
%s1: sigma 1
%s2 sigma 2
%t: teta
%nMC: numero di prove

%% obiettivo: 
% - analisi di tutto il problema confrontando gli errori empirici con le
% formule. QUINDI: 
% 1) scriviamo le formule degli errori e confrontiamo coi
% dati 

% Come cambia il valore dell'errore al variare dei parametri? 

% 2) move s2
% Il risultato è un plot con s2 su un asse e tutti gli mse sull'altro asse

% 3) Fai la stessa cosa variando il numero di dati

%% Svolgimento 

N = 14;
s1 = 1;
nMC = 2000;
t = 0;
% MSE_ML, MSE_plain, MSE_t1, MSE_t2, avML, avplain, av1, av2 = generateAndMSE(N, s1, s2, t, nMC)

% L'idea è fare due grafici: 1) s2 vs mse 2) N vs mse

s2Vett = linspace(0, 4, 41);  % Genera un vettore di 40 varianze

i = 1;
for s2 = s2Vett
    [MSE_ML(i), MSE_plain(i), MSE_t1(i), MSE_t2(i), avML(i), avplain(i), av1(i), av2(i)] = generateAndMSE(N, s1, s2, t, nMC);
    i = i+1;
end


figure(1)
plot(s2Vett, MSE_ML, col='blue')
hold on
plot(s2Vett, MSE_plain, col='red')
hold on 
plot(s2Vett, MSE_t1, col='black')
hold on 
plot(s2Vett, MSE_t2, col='green')
xlabel('S2')
ylabel('MSE')
title('S2 vs MSE')
legend({'ML', 'avg', 't1', 't2'}, 'Location', 'northeast')


Nvett = linspace(10, 150, 71);
s2 = 0.4;

i = 1;
for N = Nvett
    [MSE_MLn(i), MSE_plainn(i), MSE_t1n(i), MSE_t2n(i), avMLn(i), avplainn(i), av1n(i), av2n(i)] = generateAndMSE(N, s1, s2, t, nMC);
    i = i+1;
end

figure(2)
plot(Nvett, MSE_MLn, col='blue')
hold on
plot(Nvett, MSE_plainn, col='red')
hold on 
plot(Nvett, MSE_t1n, col='black')
hold on 
plot(Nvett, MSE_t2n, col='green')
xlabel('Number of samples')
ylabel('MSE')
title('Samples vs MSE')
legend({'ML', 'avg', 't1', 't2'}, 'Location', 'northeast')



