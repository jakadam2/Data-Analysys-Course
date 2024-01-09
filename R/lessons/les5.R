############### R Lab: Ridge and LASSO Regression ################
##################################################################
# use the "glmnet" package in order to perform ridge regression and the LASSO. We do not use the y ∼ x syntax here, but matrix and vector.
# Perform ridge regression and the lasso in order to predict Salary on the Hitters data.
# Missing values has to be removed.
# It has an alpha argument that determines what type of model is fit.
# If alpha = 0 a ridge regression model is fit.
# If alpha = 1 (default) then a LASSO model is fit. 
library(glmnet)
library(ISLR)
Hitters=na.omit(Hitters)
dim(Hitters)
x = model.matrix(Salary~., Hitters)[,-1] # without 1's
y = Hitters$Salary
# The model.matrix() function is particularly useful for creating x; not only does it produce a matrix corresponding to the 19 predictors but it also automatically transforms any qualitative variables into dummy variables.
# The latter property is important because glmnet() can only take numerical, quantitative inputs.

######### Ridge ########
# By default the glmnet() function automatically selects range of lambda values. A lambda set overrides default, a decreasing sequence of lambda values is provided from (10^10 to 10^-2 <- close to 0) 
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
# By default glmnet() standardizes the variables so that they are on the same scale. To turn off this default setting, use the argument standardize=FALSE.
# Associated with each value of lambda is a vector of ridge regression coefficients, stored in a matrix that can be accessed by coef(), in this case 20x100, 19+intercept for each lambda value:
dim(coef(ridge.mod))
# We expect the coefficient estimates to be much smaller, in terms of l2 norm, when a large value of lambda is used, as compared to when a small value is used. 
ridge.mod$lambda[50] # grid[50] = 11497.57
coef(ridge.mod)[,50] # corresponding coefficients
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # l2 norm
ridge.mod$lambda[60] # lambda = 705.48
coef(ridge.mod)[,60] # corresponding coefficients
sqrt(sum(coef(ridge.mod)[-1,60]^2)) # l2 norm > l2 for lambda[50]
# obtain the ridge regression coefficients for a new lambda, say 50:
# ?predict.glmnet #for help
predict(ridge.mod,s=50,type="coefficients")[1:20,] 

#### Validation approach to estimate test error ####
set.seed(2022)
train=sample(1:nrow(x), 0.7*nrow(x)) # another typical approach to sample
test=(-train)
y.test=y[test]
# fit a ridge regression model on the training set, and evaluate its MSE on the test set, using lambda = 4. 
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid,thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,]) # Note the use of the predict() function again. This time we get predictions for a test set, by replacing type="coefficients" with the newx argument.
mean((ridge.pred-y.test)^2) # test MSE
mean((mean(y[train ])-y.test)^2) # test MSE, if we had instead simply fit a model with just an intercept, we would have predicted each test observation using the mean of the training observations.
# We could also get the same result by fitting a ridge regression model
# with a very large value of lambda, i.e. 10^10:
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2) # like intercept only
# Least squares is simply ridge regression with lambda=0;
ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train]) # corrected according to errata (glmnet pack updated)
# In order for glmnet() to yield the exact least squares coefficients when lambda = 0, we use the argument exact=T when calling the predict() function. Otherwise, the predict() function will interpolate over the grid of lambda values used in fitting the glmnet() model, yielding approximate results. When we use exact=T, there remains a slight discrepancy in the third decimal place between the output of glmnet() when lambda = 0 and the output of lm(); this is due to numerical approximation on the part of glmnet().
mean((ridge.pred-y.test)^2)
# Compare the results from glmnet when lambda=0 with lm()
lm(y~x, subset=train)
predict(ridge.mod,s=0,exact=T,type="coefficients",x=x[train,],y=y[train])[1:20,] # corrected according to errata 
# In general, if we want to fit a (unpenalized) least squares model, then we should use the lm() function, since that function provides more useful outputs,such as standard errors and p-values.
summary(lm(y~x, subset=train))

## CROSS-VALIDATION
###################
# Instead of using the arbitrary value lambda=4, cv.glmnet() uses cross-validation to choose the tuning parameter
# By default, the function performs ten-fold cross-validation, 
# though this can be changed using the argument nfolds.
set.seed (2022)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)

dev.new(); dev.new()
plot(cv.out)
dev.print(device=pdf, "cvOut.pdf")

bestlam=cv.out$lambda.min; bestlam; log(bestlam) # the best lambda (212 on the text)
cv.out$lambda.1se # one standard error rule
log(cv.out$lambda.1se) 
ridge.pred=predict(ridge.mod,s=bestlam ,newx=x[test,])
mse_ridge <- mean((ridge.pred-y.test)^2); mse_ridge
# This represents a further improvement over the test MSE when lambda=4. 
# Finally refit our ridge regression model on the full data set with the best lambda
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]
# As expected, none of the coefficients are zero
# ridge regression does not perform variable selection!
dev.new()
plot(out,label = T, xvar = "lambda")
dev.print(device=pdf, "RidgeCoef.pdf")

#install.packages("plotmo")
library(plotmo)
dev.new()
plot_glmnet(out)
dev.print(device=pdf, "Plotmo.pdf")

######### LASSO ########
# use the argument alpha = 1 to perform lasso
lasso.mod = glmnet(x[train,], y[train], alpha=1, lambda=grid)
dev.new()
plot(lasso.mod,label = T)
dev.print(device=pdf, "LassoCoef.pdf")

dev.new()
plot(lasso.mod,label = T, xvar = "lambda")
dev.print(device=pdf, "LassoCoef_lambda.pdf")

dev.new(); plot_glmnet(lasso.mod, xvar = "lambda")
dev.print(device=pdf, "LassoCoef_names.pdf")

# perform cross-validation
set.seed (2020)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
dev.new()
plot(cv.out)
dev.print(device=pdf, "LassoMSE.pdf")

bestlam=cv.out$lambda.min; print(bestlam);print(log(bestlam))
cv.out$lambda.1se -> lambda_1se
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))
lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
mse_lasso_opt <- mean((lasso.pred-y.test)^2); mse_lasso_opt

# wrt lm
lasso.pred=predict(lasso.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
mse_lm <- mean((lasso.pred-y.test)^2); mse_lm
# However, the lasso has a substantial advantage:
# some of the 19 coefficient estimates are exactly zero (12 on the text).
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]
cat("Number of coefficients equal to 0:",sum(lasso.coef==0),"\n")
# compare with OLS when only selected predictors are included. 
#fit.lm=lm(Salary~Hits+Walks+CRuns+CRBI+League+Division+PutOuts, data=Hitters) # 12 coeffs = 0 on the textbook
#fit.lm=lm(Salary~AtBat+Hits+Walks+Years+CHmRun+CRuns+CRBI + League + Division+PutOuts+Errors, data=Hitters)
#fit.lm=lm(Salary~Hits+Walks+CRuns+CRBI + Division +PutOuts, data=Hitters)
#fit.lm=lm(Salary~Hits+Walks+CRuns+CRBI+ League + Division +PutOuts, data=Hitters)
fit.lm=lm(Salary~AtBat+Hits+Walks+Years+CHmRun+CRuns+CRBI+CWalks+ League + Division +PutOuts + Assists + Errors, data=Hitters)
coef(fit.lm)

# Compare coefficients estimated by lasso for lambda = 0 and lm
lasso.coef=predict(out,type="coefficients",s=0)[1:20,] # lm as lasso for lambda=0
lasso.coef
coef(lm(Salary~., data=Hitters)) # small differences in coefficients

###### lm full model #######
mse_full <- mean((predict(lm(Salary~., data=Hitters[train,]),newdata = Hitters[test,])-y.test)^2); mse_full
summary(lm(Salary~., data=Hitters[train,]))

lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
mse_lasso <- mean((lasso.pred-y.test)^2); mse_lasso

lasso.pred=predict(lasso.mod,s=lambda_1se ,newx=x[test,])
mse_lasso_1se <- mean((lasso.pred-y.test)^2); mse_lasso_1se
predict(lasso.mod,type="coefficients",s=lambda_1se)[1:20,]

#### BSS ####
library(leaps)
predict.regsubsets = function(object,newdata,id,...){ # ... <-> ellipsis
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
nmax <- 19
regfit.best=regsubsets(Salary~.,data=Hitters[train,], nvmax =nmax)
# test.mat=model.matrix(Salary~.,data=Hitters[test,])
# val.errors=rep(NA,nmax)
# for(i in 1:nmax){
#   coefi = coef(regfit.best,id=i)
#   pred = test.mat[,names(coefi)]%*%coefi
#   val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
# }
val.errors=rep(NA,nmax)
for(i in 1:nmax){
  pred = predict(regfit.best,Hitters[test,],id=i)
  val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
}
val.errors
cat("\nBBS best model with", which.min(val.errors),"regressors.\n")
coef(regfit.best,which.min(val.errors))

mse_bss <- min(val.errors); mse_bss

#### Stepwise regression ####
# Backward
regfit.bwd=regsubsets(Salary~.,data=Hitters[train,], nvmax =nmax, method ="backward")
val.errors=rep(NA,nmax)
for(i in 1:nmax){
  pred = predict(regfit.bwd,Hitters[test,],id=i)
  val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
}
val.errors
cat("\nBackward stepwise regression best model with", which.min(val.errors),"regressors.\n")
mse_bwd <- min(val.errors); mse_bwd
# Hybrid
regfit.hbd=regsubsets(Salary~.,data=Hitters[train,], nvmax =nmax, method ="seqrep")
val.errors=rep(NA,nmax)
for(i in 1:nmax){
  pred = predict(regfit.hbd,Hitters[test,],id=i)
  val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
}
val.errors
cat("\nHybrid stepwise regression best model with", which.min(val.errors),"regressors.\n")
mse_hbd <- min(val.errors); mse_hbd

#### Elastic Net ####
#for (alp in seq(0.1,0.9,by=0.1)) {
alp <- 0.95
enet.mod = glmnet(x[train,], y[train], alpha=alp, lambda=grid)

dev.new()
plot(enet.mod,label = T)
dev.print(device=pdf, "EnetCoef_l1.pdf")

dev.new()
plot(enet.mod,label = T, xvar = "lambda")
dev.print(device=pdf, "EnetCoef_lambda.pdf")
plot_glmnet(enet.mod, xvar = "lambda")
dev.print(device=pdf, "EnetCoef_names.pdf")
# CV
set.seed (2023)
cv.out=cv.glmnet(x[train,],y[train],alpha=alp,nfolds = 5)
dev.new()
plot(cv.out)
dev.print(device=pdf, "EnetMSE.pdf")

bestlam_enet=cv.out$lambda.min; print(bestlam_enet);print(log(bestlam_enet))
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))
enet.pred=predict(enet.mod,s=bestlam_enet ,newx=x[test,])
mse_enet <- mean((enet.pred-y.test)^2)
print(mse_enet)
#}
predict(enet.mod,type="coefficients",s=bestlam_enet)[1:20,]
