############### R Lab: stepwise ###############
library(ISLR)
Hitters=na.omit(Hitters)
dim(Hitters)
n=nrow(Hitters)

library(leaps)
regfit.full=regsubsets(Salary~.,Hitters) # BSS

####### Forward and Backward Stepwise Selection #######
# using the argument method="forward" or method="backward".
regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="forward")
summary(regfit.fwd)
# we see that using forward stepwise selection, the best one-variable model contains only CRBI, and the best two-variable model additionally includes Hits.
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="backward")
summary(regfit.bwd)
# For this data, the best one-variable through five-variable models might be identical for best subset and forward selection.
ii=5; summary(regfit.full)$outmat[ii,]==summary(regfit.fwd)$outmat[ii,]
# However, they have different best six-variable models. 
coef(regfit.full,6)
coef(regfit.fwd,6)
coef(regfit.bwd,6)
# ... and seven-variable models. 
round(coef(regfit.full,7),3)
round(coef(regfit.fwd,7),3)
round(coef(regfit.bwd,7),3)


# F. #
## Hybrid stepwise selection #####
### olsrr::ols_step_both_p based on p-value
### regsubsets(..., method ="seqrep")

startmod=lm(Salary~1,data=Hitters)
scopmod=lm(Salary~.,Hitters)
optmodAIC <- step(startmod,direction = "both", scope=formula(lm(Salary~.,Hitters)))
extractAIC(optmodAIC)
# For comparative purposes:
library(leaps)
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="backward")
dev.new()
plot(regfit.bwd,scale = "Cp")
summ_reg_bwd <- summary(regfit.bwd)
dev.print(device=pdf, "HitterStepBwd.pdf")
cat("Location of Cp min: ", which.min(summ_reg_bwd$cp),"\n Coefficients:\n")
print(coef(regfit.bwd,which.min(summ_reg_bwd$cp)))
cat("Coefficients of optmodAIC:\n")
coefficients(optmodAIC)

optmodBIC <- step(startmod,direction = "both", scope=formula(scopmod), k=log(n)) #BIC
extractAIC(optmodBIC, k=log(n))
regfit.hyb=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="seqrep")
dev.new()
plot(regfit.hyb,scale = "bic")
dev.print(device=pdf, "HitterHybrid_BIC.pdf")
summ_reg_hyb <- summary(regfit.hyb)
cat("Location of BIC min: ", which.min(summ_reg_hyb$bic),"\n Coefficients:\n")
print(coef(regfit.hyb,which.min(summ_reg_hyb$bic)))
cat("Coefficients of optmodBIC:\n")
coefficients(optmodBIC)
# library(olsrr)
# optP <- ols_step_both_p(scopmod, details = T)
# print(optP$model)

library(MASS)
optmodMASSA <- stepAIC(startmod,direction = "both", scope=formula(scopmod))
extractAIC(optmodMASSA, k=2)
optmodMASSB <- stepAIC(startmod,direction = "both", scope=formula(scopmod), k=log(n)) #BIC
extractAIC(optmodMASSB, k = log(n))
dev.new()
par(mfrow=c(2,2))
plot(optmodMASSB)
dev.print(device=pdf, "Diag_massB.pdf")

## Choosing Among Models Using the Validation Set Approach and Cross-Validation 
####### Validation Set Approach: ####
set.seed (1)
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
test=(!train)
# apply best subset selection to the training set
regfit.best=regsubsets(Salary~.,data=Hitters[train,], nvmax =19)
# make a model matrix from the test data.
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
# We will demonstrate how we use this function while performing test MSE estimation by cross-validation.


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
dev.print(device=pdf, "k-foldCV.pdf")
# We now perform best subset selection on the full data set to obtain the 10-variable model.
reg.best=regsubsets (Salary~., data=Hitters, nvmax=19)
# coef(reg.best, 11)
coef(reg.best, which.min(mean.cv.errors)) #it selects the 10-variable model (an 11-var on the textbook)
