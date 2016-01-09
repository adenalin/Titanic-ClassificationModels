train <- read.csv("~/Documents/Current Courses/PSYCH 486/Titanic/train.csv")
test <- read.csv("~/Documents/Current Courses/PSYCH 486/Titanic/test.csv")
library(rpart)
library(randomForest)
library(caret)
str(train)
str(test)

#######################################
# Model 1: Everyone dies
#######################################

test_1 <- test
test_1$Survived <- 0
model_1 <- data.frame(PassengerId = test_1$PassengerId, Survived = test_1$Survived)
write.csv(model_1, file = "model1.csv", row.names = FALSE)

# % Correct
train_1 <- train 
train_1$Predicted <- 0
train_1$Correct <- 0
train_1$Correct[train_1$Survived == train_1$Predicted] <- 1
mean(train_1$Correct) # 61.62%

# J Performance Metric
M1P <- as.factor(train_1$Predicted)
M1S <- as.factor(train_1$Survived)
sens1 <- sensitivity(data = M1P,
            reference = M1S,
            positive = "1")
spec1 <- specificity(data = M1P,
            reference = M1S,
            negative = "0")
# J = Sensitivity + Specificity − 1 measures the proportions of correctly predicted samples for both the event and nonevent groups
sens1 + spec1 - 1 
# J = 0

###### Kaggle test = 0.62679


#######################################
# Model 2: All women survived
#######################################

test_2 <- test
test_2$Survived <- 0
test_2$Survived[test_2$Sex == "female"] <- 1
model_2 <- data.frame(PassengerId = test_2$PassengerId, Survived = test_2$Survived)
write.csv(model_2, file = "model2.csv", row.names = FALSE)
nrow(model_2) # Check number of rows

# % Correct
train_2 <- train
train_2$Predicted <- 0
train_2$Predicted[train_2$Sex == "female"] <- 1
train_2$Correct <- 0
train_2$Correct[train_2$Survived == train_2$Predicted] <- 1
mean(train_2$Correct) # 78.68%

# J Performance Metric
M2P <- as.factor(train_2$Predicted)
M2S <- as.factor(train_2$Survived)
sens2 <- sensitivity(data = M2P,
            reference = M2S,
            positive = "1")
spec2 <- specificity(data = M2P,
            reference = M2S,
            negative = "0")
sens2 + spec2 - 1
# J = 0.534

##### Kaggle Test = 0.76555


########################################
# Model 3: All women & children survived
########################################

test_3 <- test
# Create range for "child" column
test_3$Child[test_3$Age < 18] <- 1
test_3$Child[test_3$Age >= 18] <- 0
test_3$Survived <- 0
# Assign "survived" if female or child
test_3$Survived[test_3$Sex == "female"] <- 1
test_3$Survived[test_3$Child == "1"] <- 1
model_3 <- data.frame(PassengerId = test_3$PassengerId, Survived = test_3$Survived)
write.csv(model_3, file = "model3.csv", row.names = FALSE)
nrow(model_3) # Check number of rows

# % Correct
train_3 <- train
train_3$Predicted <- 0
train_3$Child[train_3$Age < 18] <- 1
train_3$Child[train_3$Age >= 18] <- 0
train_3$Predicted[train_3$Sex == "female"] <- 1
train_3$Predicted[train_3$Child == "1"] <- 1
train_3$Correct <- 0
train_3$Correct[train_3$Survived == train_3$Predicted] <- 1
mean(train_3$Correct) # 77.32%

# J Performance Metric
M3P <- as.factor(train_3$Predicted)
M3S <- as.factor(train_3$Survived)
sens3 <- sensitivity(data = M3P,
                     reference = M3S,
                     positive = "1")
spec3 <- specificity(data = M3P,
                     reference = M3S,
                     negative = "0")
sens3 + spec3 - 1
# J = 0.537

##### Kaggle Test = 0.75120


#######################################
# Model 4: Random Forest
#######################################

temptest <- test
temptrain <- train
# Combine test & train data
temptest$Survived <- NA
test_4 <- rbind(temptrain, temptest)

# Fill in missing data

# Use most common embarkment spot
test_4$Embarked[c(62, 830)] <- "S"
# Turn Embarkment point categories into factors
test_4$Embarked <- factor(test_4$Embarked)

# Use median for missing fare
test_4$Fare[1044] <- median(test_4$Fare, na.rm = TRUE)

# Use decision tree to predict missing ages
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, 
                       data = test_4[!is.na(test_4$Age),], 
                       method = "anova")
test_4$Age[is.na(test_4$Age)] <- predict(predicted_age, test_4[is.na(test_4$Age),])

# Split data back into train & test sets
train_rf <- test_4[1:891,]
test_rf <- test_4[892:1309,]

# Random forest
set.seed(3)
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                          data = train_rf,
                          importance = TRUE,
                          ntree=1000)
my_prediction <- predict(my_forest, test_rf)
model_4 <- data.frame(PassengerId = test_rf$PassengerId, Survived = my_prediction)
write.csv(model_4, file = "model4.csv", row.names = FALSE)

# % Correct
train_4 <- train_rf
train_4$Predicted <- predict(my_forest, train_4)
train_4$Correct <- 0
train_4$Correct[train_4$Survived == train_4$Predicted] <- 1
mean(train_4$Correct) # 91.11%

# J Performance Metric
M4P <- as.factor(train_4$Predicted)
M4S <- as.factor(train_4$Survived)
sens4 <- sensitivity(data = M4P,
                     reference = M4S,
                     positive = "1")
spec4 <- specificity(data = M4P,
                     reference = M4S,
                     negative = "0")
sens4 + spec4 - 1
# J = 0.788

##### Kaggle Test = 0.77990


#######################################
# Model 5: Logistic Regression
#######################################

# Discarded PassengerId, Name, Ticket, Cabin
dataset_lm <- subset(train_rf,select=c(2,3,5,6,7,8,10,12))

# Regression model
log_reg <- glm(Survived ~., family = binomial(logit), data = dataset_lm)
summary(log_reg)
mod5pred <- predict(log_reg, test, type = "response")
mod5pred <- ifelse(mod5pred > 0.5, 1, 0)
model_5 <- data.frame(PassengerId = test$PassengerId, Survived = mod5pred)
model_5$Survived[is.na(model_5$Survived)] <- 0
write.csv(model_5, file = "model5.csv", row.names = FALSE)

# % Correct
train_5 <- dataset_lm
train_5$Predicted <- predict(log_reg, train_5, type = "response")
train_5$Predicted <- ifelse(train_5$Predicted > 0.5, 1, 0)
train_5$Correct <- 0
train_5$Correct[train_5$Survived == train_5$Predicted] <- 1
mean(train_5$Correct) # 80.47%

# J Performance Metric
M5P <- as.factor(train_5$Predicted)
M5S <- as.factor(train_5$Survived)
sens5 <- sensitivity(data = M5P,
                     reference = M5S,
                     positive = "1")
spec5 <- specificity(data = M5P,
                     reference = M5S,
                     negative = "0")
sens5 + spec5 - 1
# J = 0.573

### Kaggle Test = 0.77512


#######################################
# Model 6: Adena's Model
#######################################

# FIRST ATTEMPT
# Random forest omitting missing ages, embarked, SibSp, and other indexes

# Combine data
adena_test <- test
adena_train <- train
adena_test$Survived <- NA
alldata <- rbind(adena_train, adena_test)

# Fill in missing fare
alldata$Fare[1044] <- median(alldata$Fare, na.rm = TRUE) # Using median for missing fare
adena_train2 <- alldata[1:891,]
adena_test2 <- alldata[892:1309,]
no_age <- na.omit(adena_train2)

# Random Forest
set.seed(7)
forest1 <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Parch + Fare,
                          data = no_age,
                          importance = TRUE,
                          ntree = 1000)

# Predict missing ages for test data (to avoid "NA")
predict_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, 
                       data = adena_test2[!is.na(adena_test2$Age),], 
                       method = "anova")
adena_test2$Age[is.na(adena_test2$Age)] <- predict(predict_age, adena_test2[is.na(adena_test2$Age),])
forest_predict <- predict(forest1, adena_test2)
model_6 <- data.frame(PassengerId = adena_test2$PassengerId, Survived = forest_predict)
write.csv(model_6, file = "model6.csv", row.names = FALSE)

# J Performance Metric
M6P <- as.factor(predict(forest1, train_rf))
M6S <- as.factor(train$Survived)
sens6 <- sensitivity(data = M6P,
                     reference = M6S,
                     positive = "1")
spec6 <- specificity(data = M6P,
                     reference = M6S,
                     negative = "0")
sens6 + spec6 - 1
# J = 0.752

##### Kaggle Test = 0.75598 ... might be better to keep more columns

# SECOND ATTEMPT
# Neural Networks

library(nnet)
set.seed(512)
train_7 <- train
train_7$Surv <- class.ind(train_7$Survived)
titanicnn <- nnet(Surv ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                  data = train_7,
                  size = 1, 
                  softmax = TRUE,
                  decay = 5e-4, 
                  maxit = 200)
summary(titanicnn)
nnpredictions <- predict(titanicnn, test_rf)
nnpredictions <- ifelse(nnpredictions[,2] > 0.5, 1, 0)
model_7 <- data.frame(PassengerId = test_rf$PassengerId, Survived = nnpredictions)
write.csv(model_7, file = "model7.csv", row.names = FALSE)

# J Performance Metric
nncorrect <- predict(titanicnn, train_rf)
nncorrect <- ifelse(nncorrect[,2] > 0.5, 1, 0)
M7P <- as.factor(nncorrect)
M7S <- as.factor(train$Survived)
sens7 <- sensitivity(data = M7P,
                     reference = M7S,
                     positive = "1")
spec7 <- specificity(data = M7P,
                     reference = M7S,
                     negative = "0")
sens7 + spec7 - 1
# J = approximately 0.526

##### Kaggle Test: 0.76555

# THIRD ATTEMPT
# Tree with feature engineering

ctrain <- train
ctest <- test
ctest$Survived <- NA
dataset <- rbind(ctrain, ctest)

# Fill in missing embarkment points and factorize
dataset$Embarked[c(62, 830)] <- "S"
dataset$Embarked <- factor(dataset$Embarked)

# Create new "Title" column & factorize
dataset$Name <- as.character(dataset$Name)
dataset$Title <- sapply(dataset$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
dataset$Title <- sapply(dataset$Title, FUN=function(x) {strsplit(x, split=' ')[[1]][2]})
table(dataset$Title)
dataset[which(dataset$Title=="the"),"Title"] <- "Countess"
dataset$Title <- as.factor(dataset$Title)

# Convert fare to tiers
dataset$Fare[is.na(dataset$Fare)] <- median(dataset$Fare, na.rm=TRUE)
dataset$SpFare[dataset$Fare <= 20] <- 1
dataset$SpFare[dataset$Fare > 20 & dataset$Fare <= 40] <- 2
dataset$SpFare[dataset$Fare > 40 & dataset$Fare <= 60] <- 3
dataset$SpFare[dataset$Fare > 60 & dataset$Fare <= 80] <- 4
dataset$SpFare[dataset$Fare > 80 & dataset$Fare <= 100] <- 5
dataset$SpFare[dataset$Fare > 100] <- 6
dataset$SpFare <- factor(dataset$SpFare)

# Create new "Family" column
dataset$Family <- dataset$SibSp + dataset$Parch + 1

# Fill in missing ages using median age based on "Title"
titles <- unique(dataset$Title)
median.age <- c()
for (i in 1:length(titles)) {
  median.age[i] <- median(dataset$Age[dataset$Title==titles[i]], na.rm=TRUE)
}
titles.age <- as.data.frame(cbind(as.character(titles), median.age))
titles.age$median.age <- as.numeric(as.character((titles.age$median.age)))
for (i in 1:nrow(dataset)) {
  if (is.na(dataset[i,"Age"])) {
    for (j in 1:nrow(titles.age)) {
      if (titles.age[j,1]==dataset[i,"Title"]) {
        dataset$Age[i] <- titles.age[j,2]
      }
    }
  }
}

# Split back into train & test sets
fetrain <- dataset[1:891,]
fetest <- dataset[892:1309,]

# Generate decision tree
feforest <- rpart(Survived ~ Pclass + 
                           Sex + 
                           Age + 
                           SibSp + 
                           Parch + 
                           Fare + 
                           Embarked + 
                           Title +
                           SpFare +
                           Family,
                          data = fetrain,
                          method = "class")
fetest$Survived <- predict(feforest, fetest, type = "class")
model_8 <- data.frame(PassengerId = fetest$PassengerId, Survived = fetest$Survived)
write.csv(model_8, file = "model8.csv", row.names = FALSE)

# % Correct
fetrain$Predicted <- predict(feforest, fetrain, type = "class")
fetrain$Correct <- 0
fetrain$Correct[fetrain$Survived == fetrain$Predicted] <- 1
mean(fetrain$Correct) # 85.19%

# J Performance Metric
M8P <- as.factor(fetrain$Predicted)
M8S <- as.factor(fetrain$Survived)
sens8 <- sensitivity(data = M8P,
                     reference = M8S,
                     positive = "1")
spec8 <- specificity(data = M8P,
                     reference = M8S,
                     negative = "0")
sens8 + spec8 - 1
# J = 0.652

##### Kaggle Test Results: 
# 0.794 (decision tree, median ages, 6 tiers)
# 0.775 (RF, anova ages, 7 tiers)
# 0.765 (RF, median ages, 6 tiers)


# FOURTH (& BEST) ATTEMPT
# For fun... decision tree with anova ages + 7 fare tiers + feature engineering

ctrain <- train
ctest <- test
ctest$Survived <- NA
dataset <- rbind(ctrain, ctest)

# Fill in missing embarkment points and factorize
dataset$Embarked[c(62, 830)] <- "S"
dataset$Embarked <- factor(dataset$Embarked)

# Create new "Title" column & factorize
dataset$Name <- as.character(dataset$Name)
dataset$Title <- sapply(dataset$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
dataset$Title <- sapply(dataset$Title, FUN=function(x) {strsplit(x, split=' ')[[1]][2]})
table(dataset$Title)
dataset[which(dataset$Title=="the"),"Title"] <- "Countess"
dataset$Title <- as.factor(dataset$Title)

# Convert fare to tiers
dataset$Fare[is.na(dataset$Fare)] <- median(dataset$Fare, na.rm=TRUE)
dataset$SpFare[dataset$Fare <= 20] <- 1
dataset$SpFare[dataset$Fare > 20 & dataset$Fare <= 40] <- 2
dataset$SpFare[dataset$Fare > 40 & dataset$Fare <= 60] <- 3
dataset$SpFare[dataset$Fare > 60 & dataset$Fare <= 80] <- 4
dataset$SpFare[dataset$Fare > 80 & dataset$Fare <= 100] <- 5
dataset$SpFare[dataset$Fare > 100 & dataset$Fare <= 150] <- 6
dataset$SpFare[dataset$Fare > 150] <- 7
dataset$SpFare <- factor(dataset$SpFare)

# Create new "Family" column
dataset$Family <- dataset$SibSp + dataset$Parch + 1

# Fill in missing ages
age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + SpFare + Title + Family, 
                     data = dataset[!is.na(dataset$Age),], 
                     method = "anova")
dataset$Age[is.na(dataset$Age)] <- predict(age, dataset[is.na(dataset$Age),])

# Split back into train & test sets
fetrain <- dataset[1:891,]
fetest <- dataset[892:1309,]

# Decision tree
dt <- rpart(Survived ~ Pclass + 
                    Sex + 
                    Age + 
                    SibSp + 
                    Parch + 
                    Fare + 
                    Embarked + 
                    Title +
                    SpFare +
                    Family,
                  data = fetrain,
                  method = "class")
fetest$Survived <- predict(dt, fetest, type = "class")
model_9 <- data.frame(PassengerId = fetest$PassengerId, Survived = fetest$Survived)
write.csv(model_9, file = "model9.csv", row.names = FALSE)

# % Correct
fetrain$Predicted <- predict(dt, fetrain, type = "class")
fetrain$Correct <- 0
fetrain$Correct[fetrain$Survived == fetrain$Predicted] <- 1
mean(fetrain$Correct) # 83.19%

# J Performance Metric
M9P <- as.factor(fetrain$Predicted)
M9S <- as.factor(fetrain$Survived)
sens9 <- sensitivity(data = M9P,
                     reference = M9S,
                     positive = "1")
spec9 <- specificity(data = M9P,
                     reference = M9S,
                     negative = "0")
sens9 + spec9 - 1
# J = 0.635

### Kaggle Test = 0.799

#############################
##  Plot of model results  ##
#############################

library(ggplot2)
model <- c(1, 2, 3, 4, 5, 6)
kaggle <- c(0.62679, 0.76555, 0.75120, 0.77990, 0.77512,  0.79904)
ggplot() +
  geom_bar(aes(x=model, y=kaggle), 
           stat = "identity",
           width = 0.5) +
  labs(x = "Model", 
       y = "Kaggle Test Score", 
       title = "Models' Performance on Titanic Dataset")

### PRESENTATION LINK ###
### http://rpubs.com/adena/titanic