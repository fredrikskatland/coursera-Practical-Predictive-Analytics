

# Coursera Practical Predictive Analytics: Models and Methods
# Kaggle assignment

require(data.table)
require(caret)
require(Metrics)
set.seed(1)
train <- fread("/home/fredrik/R/coursera/Predicitive Analytics Models and Methods/kaggle assignment/train.csv", stringsAsFactors = F)

str(train)
# Part 1: Problem Description. 
# Give the name of the competition you selected and write a few sentences describing the competition problem as you interpreted it. 
# You want your writeup to be self-contained so your peer-reviewer does not need to go to Kaggle to study the competition description. 
# Clarity is more important than detail. What's the overall goal? What does the data look like? How will the results be evaluated?

# Description:
# According to the competition evaluation the goal is to predict the sales price for each house in the dataset based on the 
# given attributes. This is a regression problem where we're supposed to predict a numeric continous outtome, which is the sales price.
# In the kaggle competition, the evaluation of the solution to the problem is root mean square error (RMSE) 
# between the predicted sales price and the actual sales price. 
# So one can interpret the problem as a optimization problem, where we're making a regression model that minimizes the RMSE.
# Each record is a attribute vector that describes a particular home, including it's sales price (in the training set).

# Part 2: Analysis Approach. Write a few sentences describing how you approached the problem. 
# What techniques did you use? Clarity is more important than technical depth in this exercise.

# I chose R as my tool for this problem. I've used the following packages: data.table, caret, Metrics
# I did a quick exploratory analysis of the data, checking for missing values and checking for outliers by plotting the data.
# I removed the following columns because of missing values: Alley, PoolQC, Fence, MiscFeature.
# I tabulated each categorical variable to check if some of the variables had many levels which would pose a problem in the modelling stage.
# Non of the categorical variables and extremely many different values, so I didn't do anything.
# I split the data into training and testing sets (70% training and 30% testing) so I can train the model on the training set
# and evaluate the model on the testing set. 1024 observations in the training set and 436 observations in the testing set
# The dataset contain both continous variables and categorical variables. The categorical variables will need a transformation
# to fit into the regression model.
# I realized that there are a lot of variables compared to the number of record. And the number of variables will increase
# when the categorical variables are transformed to dummy variables.

sapply(train, function(x) sum(is.na(x)))
classes <- sapply(train, class)

sapply(train[,colnames(train) %in% names(classes[classes == "character"]), with = F], function(x) table(x))
train2 <- train[,!(colnames(train) %in% c("Alley","PoolQC","Fence","MiscFeature")), with = F]
# Replacing NA with 0

train2[is.na(train2)] <- 0
table(complete.cases(train2))

hist(train2$SalePrice, breaks= 100)
hist(log(train2$SalePrice), breaks= 100)

# CREATING A MODEL BASED ON SIMPLE INTUITIVE VARIABLES
# OverallQual, OverallCond, YearBuilt
train2 <- train2[complete.cases(train2) & train2$LotArea < 20000,]
plot(train2$OverallQual, log(train2$SalePrice))
plot(train2$YearBuilt, log(train2$SalePrice))
plot(train2$LotArea, log(train2$SalePrice))
plot(train2$GrLivArea, log(train2$SalePrice))
plot(train2$YearRemodAdd, log(train2$SalePrice))

colnames(train2) <- gsub("1st","first", colnames(train2))
colnames(train2) <- gsub("2nd","second", colnames(train2))
colnames(train2) <- gsub("3rd","third", colnames(train2))
colnames(train2) <- gsub("3S","threeS", colnames(train2))

train2$logSalePrice <- log(train2$SalePrice)
SalePrice <- train2$SalePrice
train2$SalePrice <- NULL


trainIdx <- createDataPartition(train2$logSalePrice, p = 0.7)

training <- data.frame(train2[trainIdx$Resample1,])
testing <- data.frame(train2[-trainIdx$Resample1,])

# 
# formula1 <- paste(colnames(train2)[-ncol(train2)],collapse = "+")
# formula2 <- paste("~-1+",formula1, sep = "")
# 
# train.onehot <- model.matrix(data = train2, eval(parse(text = formula2)))
# # removing variables with a 0-% > 0.95
# sparse <- apply(train.onehot, 2, function(x) sum(x == 0)/length(x)) >= 0.95
# names.sparse <- names(sparse[sparse == TRUE])
# train.onehot2 <- data.frame(train.onehot[,!(colnames(train.onehot) %in% names.sparse)])
# train.onehot2$logSalePrice <- train2$logSalePrice
# # REMOVING SPARSE COLUMNS
# 
# trainIdx <- createDataPartition(train2$logSalePrice, p = 0.7)
# 
# training <- data.frame(train.onehot2[trainIdx$Resample1,])
# testing <- data.frame(train.onehot2[-trainIdx$Resample1,])



# PART 3
# Initial solution: Recognizing the challenge of many variables relativ to records i decide to use a selection of variables
# that I find intuitive
# My first attempt was a simple linear regression model with a log transformation of sale price 
# with the variables OverallQual,YearBuilt,LotArea,GrLivArea,YearRemodAdd (all continous) without an intercept/constant term. 
# I used the lm()-function in R to fit a linear model.

simple.lm <- lm(data = training, logSalePrice~-1+OverallQual+YearBuilt+LotArea+GrLivArea+YearRemodAdd)
preds.lm <- predict(simple.lm, testing)
summary(simple.lm)

# PART 4
# This approach yielded a RMSE of 0.1589545 on the test set.
# All the coefficients are positive and significant. 
# I inspect the residuals by histogram and scatterlot, they look normally distributed with mean ~ 0.
# I consider this solution to be decent considering how simple it is. All coefficients were significant, so in the next iteration
# it would make sense to initially keep all the variables from this solution and add more variables
# if i decide to use the same model/algorithm. However, from a visual inspection of some of the variables i suspect
# that a non-linear model would improve the result.

resid <- preds.lm - testing$logSalePrice

rmse.lm <- rmse(testing$logSalePrice, preds.lm)
rmse.lm

# PART 5
# Nonlinear model. I decided to fit a model capable of modelling non-linear relationships.
# I decided to try multivariate adaptive regression splines using the earth package in R.
# Issues with columns starting with numbers in the formula term of the function. Replacing numbers with letters.

require(earth)
model.earth <- earth(logSalePrice~OverallQual+YearBuilt+LotArea+GrLivArea+YearRemodAdd
                     +LotFrontage+OverallCond+MasVnrArea+MasVnrArea
                     +BsmtFinSF1+BsmtUnfSF+TotalBsmtSF+firstFlrSF+secondFlrSF
                     +BsmtFullBath+BedroomAbvGr+TotRmsAbvGrd+Fireplaces+GarageYrBlt+
                       GarageCars+GarageArea+WoodDeckSF+OpenPorchSF+YrSold+MoSold, data = training)
summary(model.earth)
plotmo(model.earth)

preds.earth <- predict(model.earth, testing)
rmse(preds.earth, testing$logSalePrice)

# The model yields improved results

# Loading test-data and submitting results

test <- fread("/home/fredrik/R/coursera/Predicitive Analytics Models and Methods/kaggle assignment/test.csv")
test <- test[,!(colnames(test) %in% c("Alley","PoolQC","Fence","MiscFeature")), with = F]
# Replacing NA with 0
test[is.na(test)] <- 0

colnames(test) <- gsub("1st","first", colnames(test))
colnames(test) <- gsub("2nd","second", colnames(test))
colnames(test) <- gsub("3rd","third", colnames(test))
colnames(test) <- gsub("3S","threeS", colnames(test))


test.preds <- predict(model.earth, test)
test.preds.exp <- exp(test.preds)

results <- data.frame(Id = test$Id,
                      SalePrice = test.preds.exp)

colnames(results) <- c("Id", "SalePrice")
write.table(results, "/home/fredrik/R/coursera/Predicitive Analytics Models and Methods/kaggle assignment/results.csv", 
            sep = ",", dec = ".", row.names = F)
