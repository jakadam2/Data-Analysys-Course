%N: data size
%s1: sigma 1
%s2 sigma 2
%t: teta
%nMC: numero di prove
% s1 and s2 are variances
function[MSE_ML, MSE_plain, MSE_t1, MSE_t2, avML, avplain, av1, av2] = generateAndMSE(N, s1, s2, t, nMC)

p=s2/(s1+s2);  % definisce il parametro p per lo stimatore ml 

for ii=1:nMC  % da 1 a nMC
    
    % GENERAZIOE DEI CAMPIONI 
    % random samples generation
    x1=normrnd(t, sqrt(s1), 1, N/2); %[1, N/2] è la dimensione
    x2=normrnd(t, sqrt(s2), 1, N/2);
    x=[x1,x2];

    
    % CALCOLO DEGLI STIMATORI 
    %%% ML estimator
    tML(ii)=p*mean(x1) + (1-p)*mean(x2);  % formula dello stimatore 
    %%% plain arith,etic mean
    tplain(ii)=mean(x);   % stimatore media 
    %%% censoring approaches
    t1(ii)=mean(x1);   % stimatori che considerano i campioni di un solo tipo 
    t2(ii)=mean(x2);
    
end

% calcola la media degli stimatori 
avML=mean(tML); avplain=mean(tplain); av1=mean(t1); av2=mean(t2);
% calcola l'mse degli stimatori 
MSE_ML=mean((tML-t).^2);
MSE_plain=mean((tplain-t).^2);
MSE_t1=mean((t1-t).^2);
MSE_t2=mean((t2-t).^2);

