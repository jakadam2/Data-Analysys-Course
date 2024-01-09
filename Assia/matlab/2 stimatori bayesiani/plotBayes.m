% plot dei grafici

% I parametri sono:
% N: numero di campioni 
% sigma: *varianza* dei campioni (non deviazione standard) 
% t: il valore vero del parametro da stimare 
% nMC: numero di Monte Carlo runs

clear, clc

N = 50; 
nMC = 2000; 
muy = 0;   % Il valore da stimare 
sigmay = 2;
sigmaw = 6;


%% Grafico al variare di sigmay 

sigmayVett = linspace(0, 10, 101);

i = 1;
for sigmay=sigmayVett
    [MSE_ML(i), MSE_BAY(i)] = generateMSEBayes(N, nMC, muy, sigmay, sigmaw);
    i = i+1;
end

value = sigmaw/N;


figure(1) 
plot(sigmayVett, MSE_BAY, color='blue')
hold on
plot(sigmayVett, MSE_ML, color='red')
hold on 
yline(value, color='green');
title('MSE vs \sigma_y^2, \sigma_w^2 = 6')
xlabel('\sigma_y^2')
ylabel('MSE')
legend({'Bayes', 'ML', '\sigma_w^2/N'}, 'Location', 'northeast')


%% Grafico al variare di N 

sigmay = 0.6;
i = 1;
for N=10:200
    [MSE_ML(i), MSE_BAY(i)] = generateMSEBayes(N, nMC, muy, sigmay, sigmaw);
    i = i+1;
end


figure(2) 
loglog(10:200, MSE_BAY, color='blue')
hold on
loglog(10:200, MSE_ML, color='red')
title('MSE vs # Samples')
xlabel('# Samples')
ylabel('MSE')
legend({'Bayes', 'ML', '\sigma_w^2/N'}, 'Location', 'northeast')


%% Grafico al veirare di sigmaw

N = 50; 
nMC = 500; 
muy = 0;   % Il valore da stimare 
sigmay = 0.03;
sigmaw = 1;

clear MSE_ML;
clear MSE_BAY;

sigmawVett = linspace(0, 10, 101);

i = 1;
for sigmaw=sigmawVett
    [MSE_ML(i), MSE_BAY(i), mse] = generateMSEBayes(N, nMC, muy, sigmay, sigmaw);
    i = i+1;
end

figure(3) 
plot(sigmawVett, MSE_BAY, color='blue')
hold on
plot(sigmawVett, MSE_ML, color='red')
hold on 
yline(sigmay, color='green');
title('MSE vs \sigma_w^2, \sigma_y^2 = 0.03')
xlabel('\sigma_w^2')
ylabel('MSE')
legend({'Bayes', 'ML', '\sigma_y^2'}, 'Location', 'northeast')


%% Theroetical values 

N = 50; 
nMC = 500; 
muy = 0;   % Il valore da stimare 
sigmay = 0.03;
sigmaw = 1;

clear MSE_ML;
clear MSE_BAY;

sigmawVett = linspace(0, 10, 101);

i = 1;
for sigmaw=sigmawVett
    [MSE_ML(i), MSE_BAY(i), mse] = generateMSEBayes(N, nMC, muy, sigmay, sigmaw);
    i = i+1;
end

figure(4)
plot(sigmawVett, mse, color='red')
% hold on 
% plot(sigmawVett, MSE_BAY, color='blue')
title('Experimental MSE vs theoretical MSE')
xlabel('\sigma_w^2')
ylabel('MSE')
legend({'estimator', 'theory'}, 'Location', 'northeast')

%% Grafico rispetto ai valori teorici 
% 
% N = 50; 
% nMC = 500; 
% muy = 3;   % Il valore da stimare 
% sigmay = 2;
% sigmaw = 1;
% 
% clear MSE_ML;
% clear MSE_BAY;
% 
% a = sigmay/(sigmay*(sigmaw/N));
% theory_mse = (a^2)*(sigmaw/N) + ((a-1)^2)*sigmay;
% 
% subplot(2, 2, 1);
% plot()








