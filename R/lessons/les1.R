## R website: http://cran.r-project.org/
## Front-end: RStudio (free version is enough)
##

#### Basic commands ####  
x = c(1,3,2,5) # create vector
x

x <- c(1,3,2,5) # create vector using the <- 
x

y=seq(from=4, length=4, by=1)
y
y=seq(4,7)
y
y=4:7
y
?seq
length(x)
length(y)
x+y
x/y
ls() # var in memory
rm(x)
rm(list=ls()) # eliminate all vars in memory
ls() # var in memory

#### Matrices #### 
?matrix
x=matrix(data=c(1,2,3,4), nrow=2, ncol=2)
x=matrix(c(1,2,3,4) ,2,2)
x
matrix(c(1,2,3,4),2,2,byrow=T)
sqrt(x)
x^2 # or equivalently x*x

x%*%x # matmult - base command

crossprod(x,x) # or equivalently t(x)%*%x

tcrossprod(x,x) # or equivalently x%*%t(x)


### Random numbers #### 
set.seed(1303) # to reproduce a specific sequence of pseudo-random numbers
rnorm(50)

set.seed (3)
y=rnorm (100)
mean(y) #[1] 0.0110
var(y) #[1] 0.7329
sqrt(var(y)) #[1] 0.8561
sd(y)


# other basic commands
getwd()   #return the current directory 
setwd('/Fp/Unisa/Didattica/DataAnalysis/Corso 2023_24/Lab in R') #set the working directory
dir()  #show the files in the working directory
# help(function) or # ?function  #show the help of the function
help.search('logistic regression')  #refined search in the help
install.packages('tictoc')   #download a specific package
library(tictoc) 	#load the package

#### Create a Function #### 
myfun_sum=function(x,y) 
{x+y} # {} is the block specifying the function
myfun_sum(4,5)   #creates a function 

### Define another function
myf2 = function (x,y)
  cos(y)/(1+x^2)
myf2(2,3)
# or equivalently...
myf3 = function (x,y)
{
  z=cos(y)
  z/(1+x^2)
}
myf3(2,3)


#### Graphics #### 
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
pdf("Figure.pdf")
plot(x,y,col="green")
dev.off()
png("Figure.png")
plot(x,y,col="red")
dev.off()
dev.print(device = pdf, "FigBlack.pdf")

#
x=seq(-pi,pi,length=50)
y=x
# outer(c(1,2,3),c(1,2,3))
# tcrossprod(c(1,2,3),c(1,2,3))
# crossprod(c(1,2,3),c(1,2,3))

f=outer(x,y,function (x,y) cos(y)/(1+x^2)) #f=outer(x,y,myf2)
contour (x,y,f)
contour(x,y,f,nlevels=45,add=T)
fa=(f-t(f))/2
contour(x,y,fa,nlevels=15)
image(x,y,fa)
persp(x,y,fa)
persp(x,y,fa,theta=30)
persp(x,y,fa,theta=30,phi=20)
persp(x,y,fa,theta=30,phi=70)
persp(x,y,fa,theta=30,phi=40)

####  Bivariate Normal #### 
# install.packages("mvtnorm") # if never installed
library(mvtnorm) # to load mvtnorm package
library(MASS) # useful package "Modern Applied Statistics with S"

x=seq(-8,8,length=50)
y=x
fbn=outer(x,y,function(x,y)dmvnorm(cbind(x,y),mean = rep(1,2),sigma=matrix(c(1, .5, .5, 1), 2)))
fbn2=outer(x,y,function(x,y)dmvnorm(cbind(x,y),mean = rep(0,2),sigma=matrix(c(1, .1, .1, 1), 2)))
fbn3=outer(x,y,function(x,y)dmvnorm(cbind(x,y),mean = c(0,0),sigma=diag(2)))
fbn4=outer(x,y,function(x,y)dmvnorm(cbind(x,y),mean = c(0,0),sigma=matrix(c(1, .2, .2, 4), 2)))
dev.new()
par(mfrow=c(1,2))
contour(x,y,fbn)
persp(x,y,fbn,theta = -30, phi = 25, 
      shade = 0.75, col = "gold", expand = 0.5, r = 2, 
      ltheta = 25, ticktype = "detailed")
dev.new()
par(mfrow=c(1,2))
contour(x,y,fbn2)
persp(x,y,fbn2,theta = -30, phi = 25, 
      shade = 0.75, col = "gold", expand = 0.5, r = 2, 
      ltheta = 25, ticktype = "detailed")
dev.new()
par(mfrow=c(1,2))
contour(x,y,fbn3)
persp(x,y,fbn3,theta = -30, phi = 25, 
      shade = 0.75, col = "gold", expand = 1.5, r = 2, 
      ltheta = 25, ticktype = "detailed")
dev.new()
par(mfrow=c(1,2))
contour(x,y,fbn4)
persp(x,y,fbn4,theta = -30, phi = 25, 
      shade = 0.75, col = "gold", expand = 1.5, r = 2, 
      ltheta = 25, ticktype = "detailed")
par(mfrow=c(1,1))


#### Indexing Data #### 
A=matrix(1:16,4,4)
A[2,3]
A[c(1,3),c(2,4)]
A[1:3,2:4]
A [1:2,]
A[1,]
A[1,1:4,drop=FALSE]
A[-c(1,3),]
dim(A)

a1=array(1:24,c(2,3,4))  #creates an array of 4 2x3-matr.
dim(a1)
a1[1:2,1:2,4] #extract first 2 rows and first 2 cols from the 4th array element

x=c('a', 'b', 'c', 'd') # vector of 4 characters
which(x=='b')   #returns the pertinent index


#Lists ####  (heterogeneous collection of elements)
mylist=list( 1:5, c(TRUE, FALSE, FALSE, TRUE), c("first", "second", "third"), matrix(1:9, ncol=3, nrow=3) )
mylist[[3]]  #access the third element of the list
mylist[[3]][1] #access the first sub-element of the third element of the list


####  Dataframes #### 
mydf=data.frame(id=c(1,2,3), name=c('Mario','Giulia','Fabio'), sex=c('m','f','m'), age=c(20,24,28)) 
#creates a dataframe with heterogeneous information
mydf$age   #extracts the column “age”
mydf$age=NULL   #deletes the column "age"
mydf[1:2,]   #selects the first two rows and all the columns
mydf[-c(2,3)]  #excludes the second and third columns (variables)
mydf[mydf$sex == 'm', ]  #selects male persons
mydf[order(mydf['name']),]    #orders the dataframe by name
colnames(mydf)[2]='nome'    #renames the second column of the dataframe in "nome"
summary(mydf)  # summarized info (min, max, median etc)
str(mydf)   #type of the variables composing the dataframe
mydf$id=as.character(mydf$id)   #converts in char the column “id”
str(mydf) 


#### IF-ELSE Structure #### 
x=10
if (x<0)  {  print ("the number is negative")  }    else  {print ("the number is non negative")  }

#### For cycle #### 
vect=c(1:10) 
for  (v in vect) {print(v)}  

#### for
a<-0
library(tictoc)
tic(); for(i in 1:10^7) a[i]<-sin(2*pi*i/sqrt(2)); toc() # a<-numeric(1)
tic(); a<-numeric(10^7); for(i in 1:10^7) a[i]<-sin(2*pi*i/sqrt(2)) ; toc() # faster



#### Loading Data #### 
Auto=read.table("Auto.data") # names not recognized
Auto=read.table("Auto.data",header=T,na.strings="?")
# Auto=read.csv("Auto.csv",header=T,na.strings ="?")
head(Auto)
dim(Auto)
dim(na.omit(Auto))
# unique(unlist (lapply (Auto, function (x) which (is.na (x))))) #To find all the rows in a data frame with at least one NA
Auto=na.omit(Auto)
dim(Auto)
names(Auto)

#### Additional Graphical and Numerical Summaries #### 
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

hist(mpg)
hist(mpg, breaks=15)

dev.new()
par(mfrow=c(2,1))
hist(mpg, las=1, freq=F,xlab="MPG",ylab="", main="Istogramma MPG",col="blue", xlim=c(0,50))
lines(density(mpg),lwd=2,col="red")
rug(mpg,lwd=0.25,side=1,col="green")
boxplot(mpg, horizontal=T, xlab="MPG",ylab="", main="Box-plot MPG", ylim=c(0,50)) # notice ylim!!

plot(horsepower,mpg)
pairs(Auto) # non-numeric variable
pairs(~mpg+displacement+horsepower+weight+acceleration, data=Auto)
plot(horsepower, mpg)
identify(horsepower, mpg, name)

summary(mpg)
summary(Auto)
