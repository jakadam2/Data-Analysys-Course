#Basic
x = c(1,3,2,5) # create vector
x = 1:5
x=matrix(data=c(1,2,3,4), nrow=2, ncol=2)# create matrix from vector
x%*%x # matmult - base command
crossprod(x,x) # or equivalently t(x)%*%x
tcrossprod(x,x) # or equivalently x%*%t(x)
A=matrix(1:16,4,4)
A[2,3]
A[c(1,3),c(2,4)]
A[1:3,2:4]
A [1:2,]
A[1,]
A[1,1:4,drop=FALSE]
A[-c(1,3),]
dim(A)
a1=array(1:24,c(2,3,4))
library(ISLR)
names(Hitters)
head(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters)


#Lists 
mylist=list( 1:5, c(TRUE, FALSE, FALSE, TRUE), c("first", "second", "third"), matrix(1:9, ncol=3, nrow=3) )
mylist[[3]]  #access the third element of the list
mylist[[3]][1] #access the first sub-element of the third element of the list


#Graphics  
x=rnorm(50)
y=x+rnorm(50,mean=50,sd=.1)
cor(x,y)
plot(x,y)
lm(y~x)
abline(lm(y~x),col='red')
abline(50,0.999,col='blue')
x=rnorm(100)
y=rnorm(100)
dev.new()
plot(x,y)
plot(x,y,xlab='this is the x-axis',ylab="this is the y-axis",
     main="Plot of X vs Y")

plot(cylinders , mpg)
plot(Auto$cylinders , Auto$mpg )
attach(Auto)
plot(cylinders , mpg)
cylinders = as.factor(cylinders)
plot(cylinders, mpg)
plot(cylinders, mpg, col="red")
plot(cylinders, mpg, col="red", varwidth=T)
plot(cylinders, mpg, col="red", varwidth=T, horizontal=T)
plot(cylinders, mpg, col="red", varwidth=T, xlab="cylinders", ylab="MPG")


#Loading data
setwd('myDisk:/director/directory')
getwd()#checkikng current directory
data = read.csv('path',show_col_types = FALSE)
data = read.delim()
library(readxl)
data <- read_excel("data/Tesla Deaths.xlsx", sheet = 1)


# Simple linear regression
fit = lm(medv~lstat, data=Boston)
summary(fit)
confint(fit) # confidence interval for coefficients
predict(fit, data.frame(lstat=c(5,10,15)), interval="confidence") # confidence interval for averaged response.
predict(fit, data.frame(lstat=c(5,10,15)), interval="prediction")

fit1 = lm(medv~.-rad, data=Boston)

xx <- seq(min(lstat),max(lstat),along.with = lstat) 
ci_lin <- predict(fit,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
matplot(xx,ci_lin$fit[,2],lty=2,lwd=2, col="red", type="l", add=T)
matplot(xx,ci_lin$fit[,3],lty=2,lwd=2,col="red", type="l", add=T)
pi_lin <- predict(fit,newdata=data.frame(lstat=xx),se.fit = T,interval = "prediction")
matplot(xx,pi_lin$fit[,2],lty=2,lwd=2,col="green", type="l", add=T)
matplot(xx,pi_lin$fit[,3],lty=2,lwd=2,col="green", type="l", add=T)
legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','red','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)
corBo <- round(cor(Boston), digits =2)


#Leave-One-Out Cross-Validation
glm.fit=glm(mpg~horsepower ,data=Auto)
cv.err=cv.glm(Auto,glm.fit)
cv.err$delta
for (i in 1:5){
  glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit)$delta[1]
}

#K-fold CV
glm.fit=glm(mpg~horsepower ,data=Auto)
cv.err=cv.glm(Auto,glm.fit)
cv.err$delta
for (i in 1:5){
  glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]
}


#Best Subset Selection
library(leaps)
regfit.full=regsubsets(Salary~.,data=Hitters ,nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary)# returns R2, RSS, adjusted R2, Cp, and BIC.
# We can examine these to try to select the best overall model
reg.summary$rsq # R2 statistic increases monotonically as more variables are included.
which.min(reg.summary$rss)# it can be something else than rss


#Backward Stepwise Selection
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="backward")
summary(regfit.bwd)


#Forward Stepwise Selection
regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="forward")
summary(regfit.fwd)

# However, they have different best six-variable models. 
coef(regfit.full,6)
coef(regfit.fwd,6)
coef(regfit.bwd,6)


#Hybrid stepwise selection
regfit.hyb=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="seqrep")


#Validation Set Approach:
train=sample(c(TRUE,FALSE), size,rep=TRUE)
test=(!train)
regfit.best=regsubsets(Salary~.,data=Hitters[train,], nvmax =19)
# The model.matrix() function is used in many regression packages for building an “X” matrix from data.
test.mat=model.matrix(Salary~.,data=Hitters[test,])
# Now we run a loop, and for each size i, we extract the coefficients from regfit.best for the best model of that size, multiply them into the appropriate columns of the test model matrix to form the predictions, and compute the test MSE.
val.errors=rep(NA,19)
for(i in 1:19){
  coefi = coef(regfit.best,id=i)
  pred = test.mat[,names(coefi)]%*%coefi
  val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
}
# The best model is the one that contains which.min(val.errors) (ten in the book) variables.
val.errors; which.min(val.errors) 
coef(regfit.best,which.min(val.errors)) # This is based on training data
# there is no predict() method for regsubsets(). 
# Since we will be using this function again, we can capture our steps above and write our own predict method.
predict.regsubsets = function(object,newdata,id,...){ # ... <-> ellipsis
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}


######### Cross-Validation Approach ######## 
# create a vector that allocates each observation to one of k = 10 folds and create a matrix to store the results.
k=10
set.seed(1)
folds = sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors = matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
# write a for loop that performs cross-validation
for(j in 1:k){
  best.fit=regsubsets(Salary~., data=Hitters[folds!=j,], nvmax=19)
  for(i in 1:19){
    pred = predict(best.fit, Hitters[folds==j,], id=i)
    cv.errors[j,i] = mean((Hitters$Salary[folds==j]-pred)^2)
  }
}
# This has given us a 10×19 matrix, of which the (i,j)th element corresponds to the test MSE for the i-th cross-validation fold for the best j-variable model.
mean.cv.errors=apply(cv.errors, 2, mean); mean.cv.errors# Column average
colMeans(cv.errors) # the same. It can be faster...
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b")

#Ridge
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
summary(lm(y~x, train) 
# In general, if we want to fit a (unpenalized) least squares model, then we should use the lm() function, since that function provides more useful outputs,such as standard errors and p-values.

        
        
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

######### LASSO ########
# use the argument alpha = 1 to perform lasso
lasso.mod = glmnet(x[train,], y[train], alpha=1, lambda=grid)
dev.new()


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






