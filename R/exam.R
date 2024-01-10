
# dataset preparation 
# and split into train and test sett
data = read.csv('./RegressionDA240110.csv')
head(data)
train=sample(1:nrow(data), 0.8*nrow(data))
train_data = data[train,]
Y_train = train_data[,1]
X_train = train_data[,2:31]
test = -(train)
test_data = data[test,]
Y_test = test_data[,1]
X_test = test_data[,2:31]

#checking correliation features to detecetd relevant features
# we can see that the data seems to be mostly uncorrelated. 
# and thanks summary of OLS we see that is posiible to fit a correct model
# but part of the regressors are irrelevant,
# so it suggests some of regressors selection
install.packages('corrplot')
library(corrplot)
test.fit = lm(Y~., data=train_data)
summary(test.fit)
corrdata <- round(cor(X), digits=2)
corrplot(corrdata,method='ellipse')

# Doing the summary we can see that not all
# of the regressors are actually useflul to predict Y, beaseuse few of p values are so small


# 1) STEPWISE SELECTION  
install.packages('leaps')
library(leaps)

# creation of the models
# with a different technics forward subset selection, backward subset selectiom
# and hybrid subset selection
regfit.fwd=regsubsets(Y~.,data=train_data,nvmax=30, method ="forward")
summary(regfit.fwd)

regfit.bwd=regsubsets(Y~.,data=train_data,nvmax=30, method ="backward")
summary(regfit.bwd)

regfit.hib=regsubsets(Y~.,data=train_data,nvmax=30, method ="seqrep")
summary(regfit.hib)


# plot numero di regressori forward 
# printing number of regressors which performs the best statistic
fwd.summary=summary(regfit.fwd)
cat("Location of adj-RSq max:",which.max(fwd.summary$adjr2),"\n ")
cat("Location of Cp min:",which.min(fwd.summary$cp),"\n ")
cat("Location of BIC min:",which.min(fwd.summary$bic),"\n ")


plot(fwd.summary$adjr2 ,xlab="Number of Variables ",
     ylab="Adjusted RSq",type="l")
points(which.max(fwd.summary$adjr2),max(fwd.summary$adjr2), col="red",cex=2,pch=20)
plot(fwd.summary$cp ,xlab="Number of Variables ",ylab="Cp", type="l")
points(which.min(fwd.summary$cp ),min(fwd.summary$cp),col="red",cex=2,pch=20)
plot(fwd.summary$bic ,xlab="Number of Variables ",ylab="BIC",type="l")
points(which.min(fwd.summary$bic),min(fwd.summary$bic),col="red",cex=2,pch=20)


# plot numero di regressori backward
bwd.summary=summary(regfit.bwd)
cat("Location of adj-RSq max:",which.max(bwd.summary$adjr2),"\n ")
cat("Location of Cp min:",which.min(bwd.summary$cp),"\n ")
cat("Location of BIC min:",which.min(bwd.summary$bic),"\n ")

plot(bwd.summary$adjr2 ,xlab="Number of Variables ",
     ylab="Adjusted RSq",type="l")
points(which.max(bwd.summary$adjr2),max(bwd.summary$adjr2), col="red",cex=2,pch=20)
plot(bwd.summary$cp ,xlab="Number of Variables ",ylab="Cp", type="l")
points(which.min(bwd.summary$cp ),min(bwd.summary$cp),col="red",cex=2,pch=20)
plot(bwd.summary$bic ,xlab="Number of Variables ",ylab="BIC",type="l")
points(which.min(bwd.summary$bic),min(bwd.summary$bic),col="red",cex=2,pch=20)


# plot numero di regressori per la hybrid 
hib.summary=summary(regfit.hib)
cat("Location of adj-RSq max:",which.max(hib.summary$adjr2),"\n ")
cat("Location of Cp min:",which.min(hib.summary$cp),"\n ")
cat("Location of BIC min:",which.min(hib.summary$bic),"\n ")

plot(hib.summary$adjr2 ,xlab="Number of Variables ",
     ylab="Adjusted RSq",type="l")
points(which.max(hib.summary$adjr2),max(hib.summary$adjr2), col="red",cex=2,pch=20)
plot(hib.summary$cp ,xlab="Number of Variables ",ylab="Cp", type="l")
points(which.min(hib.summary$cp ),min(hib.summary$cp),col="red",cex=2,pch=20)
plot(hib.summary$bic ,xlab="Number of Variables ",ylab="BIC",type="l")
points(which.min(hib.summary$bic),min(hib.summary$bic),col="red",cex=2,pch=20)




# Forward selection analysis

# Observing the plots of BIC, Cp and adj-R^2, we decide to acording one standart rule
# choose 13 regressors sice the bic graph of the number of regessos
# grows fastly if less than 13 regressors are taken (even if 
# Cp and adj-R hint that less regressors cold used)

# so let's see the model with 13 regressors: 
coef(regfit.fwd,13)
dev.new()
plot(regfit.fwd,scale = "Cp")

coef(regfit.bwd,13)
dev.new()
plot(regfit.bwd,scale = "Cp")

coef(regfit.hib,13)
dev.new()
plot(regfit.hib,scale = "Cp")

# 2) RIDGE REGRESSION
# here we perform ridge regression and choose the best ratio
# by a cross validation

install.packages('glmnet')
library(glmnet)
install.packages('ISLR')
library(ISLR)
matrixTrainX = model.matrix(Y~.,train_data)[,2:31]
dim(matrixTrainX)
matrixTestX =model.matrix(Y~.,test_data)[,2:31]
cv.out=cv.glmnet(matrixTrainX, Y_train, alpha=0)
bestlambdaridge = cv.out$lambda.min

# 3) LASSO REGRESSION
# we do exaclty the same thing like in Ridge
cv.out=cv.glmnet(matrixTrainX, Y_train, alpha=1)
bestlambdalasso = cv.out$lambda.min

# 3) EVALUATE ON TEST SET
# here we make a predictions on the test set and calculate test set error

matrixTest = model.matrix(Y~.,test_data)
ridge.mod = glmnet(matrixTrainX,Y_train,alpha = 0)
lasso.mod = glmnet(matrixTrainX,Y_train,alpha = 1)
ridge.pred = predict(ridge.mod,s=bestlambdaridge,matrixTestX)
lasso.pred = predict(lasso.mod,s=bestlambdalasso,matrixTestX)
ridge.error = mean((ridge.pred - Y_test)^2)
lasso.error = mean((lasso.pred - Y_test)^2)
names(coef(regfit.fwd,13))
fwd.pred = matrixTest[,names(coef(regfit.fwd,13))] %*% coef(regfit.fwd,13)
bwd.pred = matrixTest[,names(coef(regfit.bwd,13))] %*% coef(regfit.bwd,13)
hib.pred = matrixTest[,names(coef(regfit.hib,13))] %*% coef(regfit.hib,13)
hib.error = mean((hib.pred - Y_test)^2)
bwd.error = mean((bwd.pred - Y_test)^2)
fwd.error = mean((fwd.pred - Y_test)^2)
ridge.error
lasso.error
fwd.error
bwd.error
hib.error
# the forward subset selection test error was the smallest so we choose it
names(coef(regfit.fwd,13))[2:13]
model = lm(Y~X1+X2+X3+X5+X11+X14+X17+X19+X21+X22+X23+X24,data = train_data)
summary(model)
coef(regfit.fwd,13)
