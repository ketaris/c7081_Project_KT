## HEADER ####
## Who: Katie Taris
## Student number: 224038
## What: c7081 Project
## Last edited: 2022-11-23
####


## CONTENTS ####
## 01 Setup
## 02 Classification of Quality
## 03 Classification - LDA
## 04 Classification - K-Nearest Neighbour
## 05 Classification - Support Vector Machines
## 06 Classification - Decision Trees
## 07 Regression - Model Selection
## 08 Regression - Decision Trees
## 09 Regression - Linear Model
## 10 Regression - Cross Validation of Linear Model

## 01 Setup ####
## set working directory
setwd("C:/Users/Katie/Documents/RStudio")
getwd()

## Load necessary libraries

library(BART)
library(boot)
library(class)
library(dplyr)
library(e1071)
library(gbm)
library(ISLR)
library(ISLR2)
library(leaps)
library(MASS)
library(randomForest)
library(tree)

## Import red wine data
winedata <- read.csv("winequality-red.csv")
head(winedata)
summary(winedata)

## 02 Classification of Quality ####

cor(winedata)

#4 elements showing strongest correlation are 
#vol acidity, citric acid, sulphates, & alcohol
train2 <- (winedata$density < 0.99675)
class(train2)
winedata_test <- winedata[!train2, ]
dim(winedata_test)
quality_test <- winedata$quality[!train2]

## 03 Classification - LDA ####

## LDA - Linear Discriminant Analysis 4 variables
lda.fit4 <- lda(quality ~ volatile.acidity + citric.acid + sulphates + alcohol,
               data = winedata, subset = train2)
lda.fit4
plot(lda.fit4)
lda.pred4 = predict(lda.fit4, winedata_test)
names(lda.pred4)
lda.class4 = lda.pred4$class
table(lda.class4, quality_test)
mean(lda.class4 == quality_test)
sum(lda.pred4$posterior[ , 1] >= 0.5)
sum(lda.pred4$posterior[ , 1] < 0.5)
sum(lda.pred4$posterior[ , 1] > .9)

## LDA - Linear Discriminant Analysis 2 variables
lda.fit2 <- lda(quality ~ volatile.acidity + alcohol,
               data = winedata, subset = train2)
lda.fit2
plot(lda.fit2)
lda.pred2 = predict(lda.fit2, winedata_test)
names(lda.pred2)
lda.class2 = lda.pred2$class
table(lda.class2, quality_test)
mean(lda.class2 == quality_test)
sum(lda.pred2$posterior[,1] >= 0.5)
sum(lda.pred2$posterior[,1] < 0.5)
sum(lda.pred2$posterior[ , 1] > .9)


## 04 Classification - K-Nearest Neighbours ####

## Try with 2 highest correlated variables
## Split into training and test data 
train.X <- cbind(winedata$volatile.acidity, winedata$alcohol)[train2, ]
test.X <- cbind(winedata$volatile.acidity, winedata$alcohol)[!train2, ]
train.Quality <- winedata$quality[train2]

## Perform KNN with k=1
set.seed(6)
knn.pred <- knn(train.X, test.X, train.Quality, k = 1)
table(knn.pred, quality_test)
## Calculate results
dim(winedata_test)
(5+241+105+10) / 801

## Repeat KNN with k=3
set.seed(7)
knn.pred2 <- knn(train.X, test.X, train.Quality, k = 3)
table(knn.pred2, quality_test)
mean(knn.pred2 == quality_test)

## Repeat KNN with k=10
set.seed(9)
knn.pred3 <- knn(train.X, test.X, train.Quality, k = 10)
table(knn.pred3, quality_test)
mean(knn.pred3 == quality_test)

## See if standardizing the data helps with KNN
standardized.X <- scale(winedata[,-12])

##Resplit the standardized data
test <- 1:1120
train.X2 <- standardized.X[-test, ]
test.X2 <- standardized.X[test, ]
train.Y2 <- winedata$quality[-test]
test.Y2 <- winedata$quality[test]

## Perform KNN Standardized with k=1 
set.seed(4)
knn.pred4 <- knn(train.X2, test.X2, train.Y2, k=1)
table(knn.pred4, test.Y2)
mean(test.Y2 != knn.pred4)

## Perform KNN Standardized with k=3
set.seed(5)
knn.pred5 <- knn(train.X2, test.X2, train.Y2, k=3)
table(knn.pred5, test.Y2)
mean(test.Y2 != knn.pred5)

## Perform KNN Standarized with k=6
set.seed(6)
knn.pred6 <- knn(train.X2, test.X2, train.Y2, k=6)
table(knn.pred6, test.Y2)
mean(test.Y2 != knn.pred6)

## 05 Classification - Support Vector Machines ####

## Perform SVM
set.seed(4)
train_SVM <- sample(1:nrow(winedata),1200)
x <- winedata[ ,1:11]
y <- winedata[ ,"quality"]
xtrainSVM <- x[train_SVM, ]
ytrainSVM <- y[train_SVM]
xtestSVM <- x[-train_SVM, ]
ytestSVM <- y[-train_SVM]
table(ytrainSVM)

dat <- data.frame(
  x=xtrainSVM,
  y=as.factor(ytrainSVM)
)
out <- svm(y ~ ., data = dat, kernel = "linear", cost = 10)
summary(out)
table(out$fitted, dat$y)

## Calculate training success rate
(353+336)/1200

dat.te <- data.frame(x = xtestSVM,
                     y = as.factor(ytestSVM))
pred.te <- predict(out, newdata = dat.te)
table(pred.te, dat.te$y)

## Calculate test data success rate
(134+101)/(1599-1200)

## 06 Classification - Decision Trees ####

## Quality of 7 or 8 = High 
attach(winedata)
High <- factor(ifelse (quality < 7, "No", "Yes"))
winedataDT <- data.frame(winedata, High)
tree.winedata <- tree(High ~ . -quality, winedataDT)
summary(tree.winedata)

## Plotting decision trees
plot(tree.winedata)
text(tree.winedata, pretty = 0, cex = 0.6)
tree.winedata

## Perform on training and test data
set.seed(2)
train_tree <- sample(1:nrow(winedataDT), 1100) 
winedata.test.tree <- winedataDT[-train_tree, ]
High.test <- High[-train_tree]
tree.winedata2 <- tree(High ~ . -quality,
                       winedataDT, subset = train_tree)
tree.pred <- predict(tree.winedata2, winedata.test.tree, type = "class")
table(tree.pred, High.test)
## Calculate % correct
(392+34) / (392+35+38+34)

## Prune the tree
set.seed(7)
cv.winedata <- cv.tree(tree.winedata2, FUN = prune.misclass)
names(cv.winedata)
cv.winedata #size vs. dev shows optimal # nodes

## Plot data to see optimal # nodes
par(mfrow = c(1,2))
plot(cv.winedata$size, cv.winedata$dev, type = "b",
     pch=16, col = "green")
plot(cv.winedata$k, cv.winedata$dev, type = "b",
     pch=16, col = "blue")
prune.winedata <- prune.misclass(tree.winedata2, best = 6)
par(mfrow=c(1,1))
plot(prune.winedata)
text(prune.winedata, pretty = 0)

## Test pruned tree on test data
tree.pred2 <- predict(prune.winedata, winedata.test.tree, type = "class")
table(tree.pred2, High.test)
(416+21)/(416+48+14+21)


## 07 Regression - Model Selection ####
## Best subset

regfit.full <- regsubsets(alcohol ~ . - quality, 
                          data = winedata,
                          nvmax = 10)
reg.summary <- summary(regfit.full)
reg.summary 

#plotting specs to decide which model is best
par(mfrow = c(2,2))
plot(reg.summary$rss,
     xlab = "Number of Variables",
     ylab = "RSS")
plot(reg.summary$adjr2,
     xlab = "Number of Variables",
     ylab = "Adjusted Rsq")
plot(reg.summary$cp,
     xlab = "Number of Variables",
     ylab = "Cp")
plot(reg.summary$bic, 
     xlab = "Number of Variables",
     ylab = "BIC")
which.max(reg.summary$adjr2)
which.min(reg.summary$cp)
which.min(reg.summary$bic)

## Stepwise regression - Forward
regfit.fwd <- regsubsets(alcohol ~ . -quality,
                         data = winedata,
                         nvmax = 10,
                         method = "forward")
summary(regfit.fwd)

## Stepwise regression - Backward
regfit.bwd <- regsubsets(alcohol ~ . -quality,
                         data = winedata,
                         nvmax = 10, 
                         method = "backward")
summary(regfit.bwd)

## Divide data into training and test set to 
## choose best model

set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(winedata),
                replace = TRUE)
test <- (!train)
regfit.best <- regsubsets(alcohol ~ . - quality,
                          data = winedata[train, ],
                          nvmax = 10)
test.mat <- model.matrix(alcohol ~ . - quality,
                         data = winedata[test, ])
val.errors <- rep(NA, 10)
for(i in 1:10) {
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[ ,names(coefi)] %*% coefi
  val.errors[i] <- mean((winedata$alcohol[test] - pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best, 9)

par(mfrow = c(1,1))
plot(val.errors, type = "b",
     pch = 16, col = "blue",
     main = "Val Errors Plot")

#Automating those steps
predict.regsubsets <- function(object,
                               newdata, id, ...){
  form <- as.formula(object$call[[ 2 ]])
  mat <- model.matrix(form, newdata )
  coefi <- coef (object, id = id)
  xvars <- names (coefi)
  mat [, xvars] %*% coefi
}

#Find best fit with whole set,
#not just training data
#see if best variables are different
regfit.best <- regsubsets(alcohol ~ . - quality,
                          data = winedata, nvmax = 10)
coef(regfit.best, 9)

## Choose a model using cross validation
k <- 10
n <- nrow(winedata)
set.seed(1)
folds <- sample(rep(1:k, length = n))
cv.errors <- matrix(NA, k, 10,
                    dimnames = 
                      list(NULL, paste (1:10)))
#nested for() loops
for(j in 1:k){
  best.fit <- regsubsets(alcohol ~ . -quality,
                         data = winedata[folds != j, ], nvmax = 10)
  for(i in 1:10){
    pred <- predict(best.fit, winedata[folds == j, ], id = i)
    cv.errors[j, i] <- mean((winedata$alcohol[folds == j] - pred)^2)
  }
}
mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow = c(1,1))
plot(mean.cv.errors, type = "b",
     pch = 16, col = "purple")

which.min(mean.cv.errors) 

## Perform best subset selection on full data to get best model
reg.best <- regsubsets(alcohol ~ . -quality, data = winedata,
                       nvmax = 10)
coef(reg.best, 9)

## 08 Regression - Decision trees ####
set.seed(1)
train3 <- sample(1:nrow(winedata), nrow(winedata)/2)
tree.winedata3 <- tree(alcohol ~ .-quality, winedata, subset= train3)
summary(tree.winedata3)
plot(tree.winedata3)
text(tree.winedata3, pretty = 0)

## Prune tree
cv.winedata3 <- cv.tree(tree.winedata3)
plot(cv.winedata3$size, cv.winedata3$dev, type = "b")
prune.winedata3 <- prune.tree(tree.winedata3, best = 7)
plot(prune.winedata3)
text(prune.winedata3, pretty = 0)

## Cross validation
yhat <- predict(tree.winedata3, newdata = winedata[-train3, ])
winedata.test.tree2 <- winedata[-train3, "alcohol"]
plot(yhat, winedata.test.tree2)
abline(0,1)
mean((yhat - winedata.test.tree2)^2) #need sqrt(MSE)

## Bagging
set.seed(1)
bag.winedata <- randomForest(alcohol ~ . -quality,
                             data = winedata,
                             subset = train3, mtry = 10,
                             importance = TRUE)
yhat.bag <- predict(bag.winedata, newdata =  winedata[-train3, ])
plot(yhat.bag, winedata.test.tree2)
abline(0,1)
mean((yhat.bag - winedata.test.tree2)^2)

## Change number of trees
bag.winedata2 <- randomForest(alcohol ~ . -quality,
                              data = winedata,
                              subset=train3,
                              mtry=10, 
                              ntree=25)
yhat.bag2 <- predict(bag.winedata2, newdata = winedata[-train3, ])
mean((yhat.bag2 - winedata.test.tree2)^2)

## Testing growing random forest
set.seed(1)
rf.winedata <- randomForest(alcohol ~ . -quality,
                            data = winedata,
                            subset = train3,
                            mtry = 6,
                            importance = TRUE)
yhat.rf <- predict(rf.winedata, newdata = winedata[-train3, ])
mean((yhat.rf - winedata.test.tree2)^2)

importance(rf.winedata)
varImpPlot(rf.winedata)

## Boosting

set.seed(1)
boost.winedata <- gbm(alcohol ~ . -quality,
                      data = winedata[train3, ],
                      distribution = "gaussian",
                      n.trees = 5000,
                      interaction.depth = 4)
summary(boost.winedata)

## Use boosted model to predict alcohol on test set
yhat.boost <- predict(boost.winedata,
                      newdata = winedata[-train3, ],
                      n.trees = 5000)
mean((yhat.boost - winedata.test.tree2)^2)

boost.winedata2 <- gbm(alcohol ~ . -quality,
                       data = winedata[train3, ],
                       distribution = "gaussian",
                       n.trees = 5000,
                       interaction.depth = 4, 
                       shrinkage = 0.2,
                       verbose = F)
yhat.boost2 <- predict(boost.winedata2,
                       newdata = winedata[-train3, ],
                       n.trees = 5000)
mean((yhat.boost2 - winedata.test.tree2)^2)

## Bayesian additive regression trees
x <- winedata[ ,1:10]
y <- winedata[ ,"alcohol"]
xtrain <- x[train3, ]
ytrain <- y[train3]
xtest <- x[-train3, ]
ytest <- y[-train3]
set.seed(1)
bartfit <- gbart(xtrain, ytrain, x.test = xtest)

## Find test error
yhat.bart <- bartfit$yhat.test.mean
mean((ytest - yhat.bart)^2)

## Check for variable appearances
ord <- order(bartfit$varcount.mean, decreasing = T)
bartfit$varcount.mean[ord]
 
## 09 Regression - Linear Model ####
## Simple linear model using density
lm_density <- lm(alcohol ~ density, data = winedata)
lm_density
par(mfrow = c(2,2))
plot(lm_density)
par(mfrow = c(1,1))
summary(lm_density)
plot(x= winedata$density, y = winedata$alcohol,
     xlab = "Density",
     ylab = "Alcohol", 
     main = "Alcohol as a Function of Density")
abline(lm_density, col = "red")

## Take log of density
lm_density_log <- lm(alcohol ~ log(density), data = winedata)
plot(x=log(winedata$density), y = winedata$alcohol)
abline(lm_density_log, col = "red") 
summary(lm_density_log)

## Multiple regression - multiple variables in linear model
## Try using density and residual sugar
lm_sugar_den <- lm(alcohol ~ residual.sugar + density, data = winedata)
summary(lm_sugar_den)
## Try using density, residual sugar and citric acid
lm_dsc <- lm(alcohol ~ density + residual.sugar + citric.acid, data = winedata)
summary(lm_dsc)
## Try using all variables
lm_all <- lm(alcohol ~ . - quality, data = winedata)
summary(lm_all)
## Diagnostic plots
par(mfrow = c(2,2))
plot(lm_all)
par(mfrow = c(1,1))
## Try with all except free sulfur dioxide
lm_9v <- lm(alcohol ~ . - quality -free.sulfur.dioxide, data = winedata)
summary(lm_9v)

## Test lm on 5 variables
lm.5v <- lm(alcohol ~ density + fixed.acidity + residual.sugar + pH + sulphates , data = winedata)
summary(lm.5v)

## Test interactions with density
lm_denint2 <- lm(alcohol ~ density*residual.sugar, 
                      data = winedata)
summary(lm_denint2)
lm_denint3 <- lm(alcohol ~ density*citric.acid, 
                 data = winedata)
summary(lm_denint2)

## Test polynomial terms with density
lm_den2 <- lm(alcohol ~ density + I(density^2), data = winedata)
summary(lm_den2)

## Anova to compare the models
anova(lm_density, lm_den2)

## Higher order polynomials
lm_den6 <- lm(alcohol ~ poly(density, 6), data = winedata)
summary(lm_den6)


## 10 Regression - Cross Validation of Linear Model ####
## Create training and test data
set.seed(5)
trainCV <- sample(1599, 800)
lm.cv <- lm(alcohol ~ density, data = winedata, subset = trainCV)

## Use predict on test data
attach(winedata)
mean((alcohol - predict(lm.cv, winedata))[-trainCV]^2)

## Test CV on poly data
## Poly2
lm.cv2 <- lm(alcohol ~ poly(density, 2),
             data = winedata, subset = trainCV)
mean((alcohol - predict(lm.cv2, winedata))[-trainCV]^2)

## Poly3
lm.cv3 <- lm(alcohol ~ poly(density, 3),
             data = winedata, subset = trainCV)
mean((alcohol - predict(lm.cv3, winedata))[-trainCV]^2)

## Leave-one-out CV
glm.fit <- glm(alcohol ~ density, data = winedata)
cv.err <- cv.glm(winedata, glm.fit)
cv.err$delta

## Test on higher order polynomials
cv.error <- rep(0, 5)
for (i in 1:5){
  glm.fit <- glm(alcohol ~ poly(density, i), data = winedata)
  cv.error[i] <- cv.glm(winedata, glm.fit)$delta[1]
}
cv.error
plot(cv.error, type='b')

## Test CV on full model
lm.cv4 <- lm(alcohol ~ . -quality, data = winedata, subset = trainCV)
attach(winedata)
mean((alcohol - predict(lm.cv4, winedata))[-trainCV]^2)

## Test CV on 5 variable model
lm.cv5 <- lm(alcohol ~ density + fixed.acidity + residual.sugar + pH + sulphates , data = winedata, subset = trainCV)
attach(winedata)
mean((alcohol - predict(lm.cv5, winedata))[-trainCV]^2)
