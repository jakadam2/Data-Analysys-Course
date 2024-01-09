##############  R Lab: Linear regression  ##############
##############  Empirical linear models   ##############

############# Simple linear regression ################
## Boston Data
library(MASS)
?Boston
names(Boston)
dev.new()
pairs(Boston)
# dev.print(device=pdf, "pairsBoston.pdf")
fit = lm(medv~lstat, data=Boston)
summary(fit)
confint(fit) # confidence interval for coefficients
predict(fit, data.frame(lstat=c(5,10,15)), interval="confidence") # confidence interval for averaged response.
predict(fit, data.frame(lstat=c(5,10,15)), interval="prediction")

# plot the fitted line
attach(Boston)
dev.new()
plot(lstat, medv)
abline(fit,col="red",lwd=4)
dev.print(device=pdf, "medv_lstat.pdf")

xx <- seq(min(lstat),max(lstat),along.with = lstat) 
ci_lin <- predict(fit,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
matplot(xx,ci_lin$fit[,2],lty=2,lwd=2, col="red", type="l", add=T)
matplot(xx,ci_lin$fit[,3],lty=2,lwd=2,col="red", type="l", add=T)
pi_lin <- predict(fit,newdata=data.frame(lstat=xx),se.fit = T,interval = "prediction")
matplot(xx,pi_lin$fit[,2],lty=2,lwd=2,col="green", type="l", add=T)
matplot(xx,pi_lin$fit[,3],lty=2,lwd=2,col="green", type="l", add=T)
legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','red','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)

dev.print(device=pdf, "medv_lstat_conf_pred_int.pdf")

# model diagnositic plots
dev.new()
par(mfrow=c(2,2))
plot(fit)


############## Multiple Linear Regression #############
fit = lm(medv~lstat+age, data=Boston)
summary(fit)
# fit a linear model with all predictors
fit = lm(medv~., data=Boston)
summary(fit)
# check collinearity
library(car)
vif(fit) # rad and tax have high VIF values, indicating they are highly correlated with at least one of the other predictors
# look at correlation matrix;
corBo <- round(cor(Boston), digits =2)

#install.packages("corrplot") # se non installato in R
library(corrplot)
dev.new()
corrplot(corBo,method = 'ellipse')
dev.new()
corrplot.mixed(corBo,order='original',number.cex=1, upper="ellipse")
library(psych)
dev.new()
corPlot(corBo, cex = 1.1, show.legend=TRUE, main="Correlation plot")

# remove rad;
fit1 = lm(medv~.-rad, data=Boston)
summary(fit1)

summary(lm(medv~.-rad-indus-age-tax, data=Boston))


############### Interaction Terms ############### 
fit = lm(medv~lstat*age, data=Boston); summary(fit) # shorthand for
fit = lm(medv~lstat+age+lstat:age, data=Boston); summary(fit)

## Non-linear Transformations of the Predictors
fit1 = lm(medv~lstat, data=Boston); summary(fit1)
fit2 = lm(medv~lstat+I(lstat^2), data=Boston); summary(fit2)

xx <- seq(min(lstat),max(lstat),along.with = lstat)
ci_lin2 <- predict(fit2,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
pi_lin2 <- predict(fit2,newdata=data.frame(lstat=xx),se.fit = T,interval = "prediction")

par(mfrow=c(1,1))
dev.new()
plot(lstat, medv,ylim=c(0,60))
matplot(xx,ci_lin2$fit[,1],lty=1, ltw=4, col="red", type="l",add=T)
matplot(xx,ci_lin2$fit[,2],lty=2, ltw=2, col="red", type="l",add=T)
matplot(xx,ci_lin2$fit[,3],lty=2, ltw=2, col="red", type="l",add=T)
matplot(xx,pi_lin2$fit[,2],lty=2, ltw=2, col="green", type="l",add=T)
matplot(xx,pi_lin2$fit[,3],lty=2, ltw=2, col="green", type="l",add=T)
legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','red','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)

dev.print(device=pdf, "medv_lstat2_int.pdf")

###############  check model ############### 
dev.new()
par(mfrow=c(2,2))
plot(fit2)

# polynomials
fit3 = lm(medv~poly(lstat, 3), data=Boston); summary(fit3)
ci_lin3 <- predict(fit3,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
pi_lin3 <- predict(fit3,newdata=data.frame(lstat=xx),se.fit = T,interval = "prediction")

dev.new()
par(mfrow=c(1,1))
plot(lstat, medv,ylim=c(0,60))
matplot(xx,ci_lin3$fit[,1],lty=1, ltw=4, col="red", type="l", add=T)
matplot(xx,ci_lin3$fit[,2],lty=2, ltw=2, col="red", type="l", add=T)
matplot(xx,ci_lin3$fit[,3],lty=2,ltw=2, col="red", type="l", add=T)
matplot(xx,pi_lin3$fit[,2],lty=2,ltw=2, col="green", type="l", add=T)
matplot(xx,pi_lin3$fit[,3],lty=2,ltw=2, col="green", type="l", add=T)
legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','red','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)
dev.print(device=pdf, "medv_lstat3_int.pdf")

fit5 = lm(medv~poly(lstat, 5), data=Boston); summary(fit5)
lin5 <- predict(fit5,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
matplot(xx,lin5$fit[,1],lty=1, ltw=4, col="blue", type="l", add=T)

dev.print(device=pdf, "medv_lstat5_int.pdf")

dev.new()
par(mfrow=c(2,2))
plot(fit3)

# log transformation
fit4 = lm(medv~log(lstat), data=Boston); summary(fit4)
lin4 <- predict(fit4,newdata=data.frame(lstat=xx),se.fit = T,interval = "confidence")
pi_lin4 <- predict(fit4,newdata=data.frame(lstat=xx),se.fit = T,interval = "prediction")
dev.new()
par(mfrow=c(1,1))
plot(lstat, medv,ylim=c(0,60))
matplot(xx,lin4$fit[,1],lty=1, ltw=4, col="blue", type="l", add=T)
matplot(xx,lin4$fit[,2],lty=2, ltw=2, col="red", type="l", add=T)
matplot(xx,lin4$fit[,3],lty=2, ltw=2, col="red", type="l", add=T)
matplot(xx,pi_lin4$fit[,2],lty=2, ltw=2,col="green", type="l", add=T)
matplot(xx,pi_lin4$fit[,3],lty=2,ltw=2, col="green", type="l", add=T)

legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','blue','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)
dev.print(device=pdf, "medv_lstat4_int.pdf")

dev.new()
par(mfrow=c(2,2))
plot(fit4)
dev.print(device=pdf, "diag4.pdf")

############### Qualitative Predictors ############### 
# Carseats Data 
# install.packages("ISLR")
library(ISLR)
names(Carseats)
head(Carseats)
fit = lm(Sales~.+Income:Advertising+Price:Age, data=Carseats)
summary(fit)
contrasts(Carseats$ShelveLoc)


############### Simulate Data ################
## Suppose true model: y=f(x)+epsilon, where f(x)=2+3x+x^3
## where x~Uniform(-3,3), y|x~N(f(x), 2^2), i.e. epsilon~N(0, 2^2).

set.seed(2020) # In this way the results will be reproducible

f = function(x){
  return(2+3*x+x^3)
}
vv = 4 # error variance
n = 100 
x = runif(n, -3, 3)
y = rep(0, n)
for(i in 1:n){
  y[i] = rnorm(1, f(x[i]), sqrt(vv)) # N(f(x),sd) 
}
#  y=f(x)+rnorm(n,0,sqrt(vv)) # more efficient

d = data.frame(feature=x, response=y) ## create data frame
dev.new() # new window
dev.new()
plot(d$feature, d$response) ## scatter plot
xx = seq(-3, 3, length.out = 300)
lines(xx, f(xx),col="blue",lwd=3) ## true curve
## fit a simple linear model 
fit1 = lm(response~feature, data=d)
fhat_of_xx = (cbind(1, xx)%*%fit1$coefficients)
lines(xx, fhat_of_xx, col="red",lwd=3) ## fitted line
legend('topleft', c('points','true curve','fitted line'), lty=c(NA,1,1), col=c('black','blue',"red"), pch=c(1,NA,NA),lwd=c(NA,2,2), cex=1)
dev.print(device=pdf, "x3sim.pdf")


## split data into training and test
indx = sample(n, n/2)
d_train = d[indx,]
d_test = d[-indx,]
## calcuate training and test MSEs by fitting polynomial of order 1, ..., poly.order
poly.order = 10
RR <- TRUE # Set FALSE for orthogonal polynomials
mse_train = rep(NA, poly.order) # save train MSE to a vector
mse_test = rep(NA, poly.order) # save test MSE to a vector
for(i in 1: poly.order){
  fit = lm(response~poly(feature,i, raw=RR), data=d_train) ## see ?poly() how it is defined
  mse_train[i]=mean((d_train$response-fit$fitted.values)^2)
  predicted_test = predict(fit, newdata = d_test); ## predicted values on test data
  mse_test[i]=mean((d_test$response-predicted_test)^2)
}
poly_order=1:poly.order
dev.new()
plot(poly_order, mse_train, "l", lwd=4,xlim=c(0,11),ylim=c(0,30), xlab = "Polynomial degree", ylab = "MSE")
lines(poly_order, mse_test, lty=2, lwd=4)
abline(a=vv,b=0, lty=3,lwd=2)
legend("topright",legend=c("Train MSE", "Test MSE", "Irreducible error"),  col=c("black", "black", "black"), lty=c(1,2,3), lwd = c(4,4,2), cex=0.9)
dev.print(device=pdf, "PolySim1.pdf")

## find the polynomial order that yields the smallest mse
which.min(mse_train)
which.min(mse_test)

#### repeat above for many times and take the average
sim.repeat = function(nsim = 50){
  poly.order = 10
  mat.test = matrix(NA, nsim, poly.order)
  mat.train = matrix(NA, nsim, poly.order)
  for(sim in 1:nsim){
    f = function(x){
      return(2+3*x+x^3)
    }
    n = 100 
    x = runif(n, -3, 3)
    y = rep(0, n)
    for(i in 1:n){
      y[i] = rnorm(1, f(x[i]), sqrt(vv))
    }
    d = data.frame(feature=x, response=y) ## create data frame
    #plot(d$feature, d$response) ## scatter plot
    xx = seq(-3, 3, length.out = 300)
    #lines(xx, f(xx)) ## true curve
    ## fit a simple linear model 
    fit1 = lm(response~feature, data=d)
    fhat_of_xx = (cbind(1, xx)%*%fit1$coefficients)
    #lines(xx, fhat_of_xx, col="red") ## fitted line
    ## split data into training and test
    indx = sample(n, n/2)
    d_train = d[indx,]
    d_test = d[-indx,]
    ## calculate training and test MSEs by fitting polynomial of order 1, ..., poly.order
    mse_train = rep(NA, poly.order) # save train MSE to a vector
    mse_test = rep(NA, poly.order) # save test MSE to a vector
    for(i in 1: poly.order){
      fit = lm(response~poly(feature,i, raw=RR), data=d_train); ## see ?poly() how it is defined
      mse_train[i]=mean((d_train$response-fit$fitted.values)^2)
      predicted_test = predict(fit, newdata = d_test); ## predicted values on test data
      mse_test[i]=mean((d_test$response-predicted_test)^2)
    }
    mat.train[sim, ] = mse_train
    mat.test[sim, ] = mse_test
    ## find the polynomial order that yields the smallest mse
    #which.min(mse_train)
    #which.min(mse_test)
  }
  poly_order=1:poly.order
  dev.new()
  plot(poly_order, colMeans(mat.train), "l", lwd=4,xlim=c(0,11),ylim=c(0,30), xlab = "Polynomial degree", ylab = "MSE")
  lines(poly_order, colMeans(mat.test), lty=2, lwd=4)
  abline(a=vv,b=0, lty=3, lwd=2)
  legend("top",legend=c("Train MSE", "Test MSE", "Irreducible error"),  col=c("black", "black", "black"), lty=c(1,2,3), lwd = c(4,4,2), cex=0.9)
  cat("\nPolynomial degree with minimum MSE_train:",which.min(colMeans(mat.train)),"\n")
  cat("\nPolynomial degree with minimum MSE_test:",which.min(colMeans(mat.test)),"\n")
}

library(tictoc)
tic()
sim.repeat()
toc()


############### R Lab: Resampling Methods ###############
# The Auto Data is part of the library ISLR 
library(ISLR)
n = nrow(Auto) # sample size
dev.new()
plot(Auto$mpg~Auto$horsepower)

############ Leave-One-Out Cross-Validation ############ 
# The LOOCV estimate can be automatically computed for any generalized linear model using the glm() and cv.glm() functions
# The glm() function can be used to perform logistic regression by passing in the family="binomial" argument.
# But if we use glm() to fit a model without passing in the family argument, then it performs linear regression, just like the lm() function. 
glm.fit=glm(mpg~horsepower, data=Auto)
coef(glm.fit)
lm.fit=lm(mpg~horsepower ,data=Auto)
coef(lm.fit)
# we will perform linear regression using the glm() function rather than the lm() function because the former can be used together with cv.glm(). cv.lm in "lmvar" [check the differences in the syntax]
library(boot)
?cv.glm
glm.fit=glm(mpg~horsepower ,data=Auto)
cv.err=cv.glm(Auto,glm.fit)
cv.err$delta # The LOOCV estimate for the test error is approximately 24.23.
# LOOCV for polynomial regressions with orders i=1,2,...,5.
cv.error=rep(0,5)
library(tictoc)
tic()
for (i in 1:5){
  glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit)$delta[1]
}
cat("LOOCV errors: ",cv.error,"\n")
toc()
# we see a sharp drop in the estimated test MSE between the linear and quadratic fits, but then no clear improvement from using higher-order polynomials.

attach(Auto)
glm.fit=glm(mpg~poly(horsepower ,2),data=Auto)
xx <- seq(min(horsepower),max(horsepower),along.with = horsepower)
lin2 <- predict(glm.fit,newdata=data.frame(horsepower=xx),se.fit = T,interval = "confidence")


par(mfrow=c(1,1))
dev.new()
plot(Auto$mpg~Auto$horsepower,lwd=2,cex.lab=1.5,cex.axis=1.2)
matplot(xx,lin2$fit,lty=1, ltw=2, lwd=3, col="red", type="l", add=T)


############## k-Fold Cross-Validation ##############
# The cv.glm() function can also be used to implement k-fold CV.
set.seed(17)
cv.error.10 = rep(0,10)
tic()
for(i in 1:10){
  glm.fit = glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]
}
cat("10-fold CV errors: ",cv.error.10,"\n")
toc()
# Notice that the computation time is much shorter than that of LOOCV.
# We still see little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.

