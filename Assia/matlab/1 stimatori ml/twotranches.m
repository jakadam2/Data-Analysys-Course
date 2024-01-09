    %N: data size
%s1: sigma 1
%s2 sigma 2
%t: teta
%nMC: numero di prove
function[MSE_ML, MSE_plain, MSE_t1, MSE_t2, avML, avplain, av1, av2] = twotranches(N, s1, s2, t, nMC)

% s1 and s2 are variances

p=s2/(s1+s2)  % definisce il parametro p per lo stimatore ml 

for ii=1:nMC  % da 1 a nMC
    % random samples generation
    x1=normrnd(t, sqrt(s1), 1, N/2); %[1, N/2] è la dimensione
    x2=normrnd(t, sqrt(s2), 1, N/2);
    x=[x1,x2];
    %%% ML estimator
    tML(ii)=p*mean(x1) + (1-p)*mean(x2)  % formula dello stimatore 
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

figure()

% Grafico con 4 "finestre"

subplot(4,1,1)  
plot(1:nMC,tML, '-bo',1:nMC,t*ones(1,nMC),'--r',...
    'markersize', 10, 'markerface','g','linewidth', 2)
xlabel('MC runs', 'fontsize', 18)
ylabel('ML values','fontsize', 18)
axis([1 nMC t-1 t+2]);
grid

subplot(4,1,2) 
plot(1:nMC,tplain, '-bo',1:nMC,t*ones(1,nMC),'--r',...
    'markersize', 10, 'markerface','g','linewidth', 2)
xlabel('MC runs', 'fontsize', 18)
ylabel('arithmetic average','fontsize', 18)
axis([1 nMC t-1 t+2]);
grid

subplot(4,1,3) 
plot(1:nMC,t1, '-bo',1:nMC,t*ones(1,nMC),'--r',...
    'markersize', 10, 'markerface','g','linewidth', 2)
xlabel('MC runs', 'fontsize', 18)
ylabel('only tranche 1','fontsize', 18)
axis([1 nMC t-1 t+2]);
grid

subplot(4,1,4) 
plot(1:nMC,t2, '-bo',1:nMC,t*ones(1,nMC),'--r',...
    'markersize', 10, 'markerface','g','linewidth', 2)
xlabel('MC runs', 'fontsize', 18)
ylabel('only tranche 2','fontsize', 18)
axis([1 nMC t-1 t+2]);
grid