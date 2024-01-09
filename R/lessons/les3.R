#################   R Lab: BSS   #################
### The Hitters data will be used in this lab #####
# We wish to predict a baseball player’s Salary on the basis of various statistics associated with performance in the previous year. First of all, we note that the Salary variable is missing for some of the players 
library(ISLR)
names(Hitters)
head(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))
n = nrow(Hitters); n # sample size

################### Subset Selection #######################
######### Best Subset Selection ######## 
# The regsubsets() function (part of the leaps library) performs best subset selection by identifying the best model that contains a given number of predictors, where best is quantified using RSS.
# Same syntax of lm()
library(leaps)
regfit.full=regsubsets(Salary~.,Hitters)
# An asterisk indicates that a given variable is included in the corresponding model
summary(regfit.full)
# the nvmax option can be used in order to return as many variables as are desired. 
regfit.full=regsubsets(Salary~.,data=Hitters ,nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary)# returns R2, RSS, adjusted R2, Cp, and BIC.
# We can examine these to try to select the best overall model
reg.summary$rsq # R2 statistic increases monotonically as more variables are included.
which.min(reg.summary$rss) ## identify the location of the minimum
cat("\nLocation of RSS min:",which.min(reg.summary$rss),"\n")
cat("Location of adj-RSq max:",which.max(reg.summary$adjr2),"\n ")
cat("Location of Cp min:",which.min(reg.summary$cp),"\n ")
cat("Location of BIC min:",which.min(reg.summary$bic),"\n ")
# Plot RSS, adjusted R2, Cp, and BIC for all of the models at once
dev.new()
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",
     type="l")
points(which.min(reg.summary$rss),min(reg.summary$rss), col="red",cex=2,pch=20)
plot(reg.summary$adjr2 ,xlab="Number of Variables ",
     ylab="Adjusted RSq",type="l")
points(which.max(reg.summary$adjr2),max(reg.summary$adjr2), col="red",cex=2,pch=20)
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type="l")
points(which.min(reg.summary$cp ),min(reg.summary$cp),col="red",cex=2,pch=20)
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type="l")
points(which.min(reg.summary$bic),min(reg.summary$bic),col="red",cex=2,pch=20)
dev.print(device=pdf, "HitterBSS.pdf")
# The regsubsets() function has a built-in plot() command which can be used to display the selected variables for the best model with a given number of predictors, ranked according to the R2, BIC, Cp, adjusted R2. 
# To find out more about this function, type ?plot.regsubsets.
dev.new()
plot(regfit.full,scale="r2")
dev.print(device=pdf, "HitterBSS_R2.pdf")
dev.new()
plot(regfit.full,scale="adjr2")
dev.print(device=pdf, "HitterBSS_adjR2.pdf")
dev.new()
plot(regfit.full,scale="Cp")
dev.print(device=pdf, "HitterBSS_Cp.pdf")
dev.new()
plot(regfit.full,scale="bic")
dev.print(device=pdf, "HitterBSS_BIC.pdf")
# The top row of each plot contains a black square for each variable selected according to the optimal model associated with that statistic.
coef(regfit.full ,which.min(reg.summary$bic)) #see the coefficient estimates for the 6-variable model
