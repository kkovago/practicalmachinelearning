
# load libraries
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(rpart)

# download data if ther is no local data file yet
sourceDirURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
trainingFileName <- "pml-training.csv"
testFileName <- "pml-testing.csv"

downloadData <- function(sourceURL, fileName) {
    if(!file.exists(fileName)) 
        download.file(paste0(sourceURL, fileName),fileName)  
}

# load data
downloadData(sourceDirURL, trainingFileName)
downloadData(sourceDirURL, testFileName)

trainingSet <- read.csv(trainingFileName, na.strings=c("NA","#DIV/0!",""))
testSet <- read.csv(testFileName, na.strings=c("NA","#DIV/0!",""))

# step 1: checking NA
as.factor(apply(trainingSet, 2, function(x) sum(is.na(x)) ))[2]
# 17 Levels: 0 19216 19217 19218 19220 19221 19225 19226 19227 ... 19622
# All those columns that have NA, have NA in over 90% of their rows 
# so I will delete columns with any NA
# first find the columns with no NA
columnsNoNA <- apply(trainingSet, 2, function(x) {sum(is.na(x)) == 0} )
#then filter the tyraining set to those columns only
trainingSet <- trainingSet[, columnsNoNA]

# step 2: remove zero covariates
trainingSet <- trainingSet[, nearZeroVar(trainingSet, saveMetrics = TRUE)$nzv==FALSE]

# step 3: The first 5 columns have no value (user and timestamp data), so delete them.
# "X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2","cvtd_timestamp"
trainingSet <- trainingSet[,-c(1:5)]
dim(trainingSet)

# Create cross validation set
set.seed(2017)
inTrain <- createDataPartition(trainingSet$classe, p=0.6, list=FALSE)
cleanTrainingSet <- trainingSet[inTrain, ]
cleanValidationSet <- trainingSet[-inTrain, ]

# Examine "classe"
plot(cleanTrainingSet$classe, col="blue", 
     main="Frequency of 'classe' values", xlab="'classe' values", ylab="Frequency")

# Generate training models for random forest, decision tree and gbm methods
# As this takes a lot of time, save the model when first calculated
# then use the saved model for subsequent calculations
if(file.exists("gbm.trainingModel.RData")) {
  load(file="gbm.trainingModel.RData", verbose=TRUE)
} else {
  gbm.trainingModel <- train(classe ~ ., data=cleanTrainingSet, method="gbm")
  save(gbm.trainingModel, file="gbm.trainingModel.RData")
}

if(file.exists("rf.trainingModel.RData")) {
  load(file="rf.trainingModel.RData", verbose=TRUE)
} else {
  rf.trainingModel <- train(classe ~ ., data=cleanTrainingSet, method="rf")
  save(rf.trainingModel, file="rf.trainingModel.RData")
}

if(file.exists("rpart.trainingModel.RData")) {
  load(file="rpart.trainingModel.RData", verbose=TRUE)
} else {
  rpart.trainingModel <- train(classe ~ ., data=cleanTrainingSet, method="rpart")
  save(rpart.trainingModel, file="rpart.trainingModel.RData")
}

# make the predictions on the validation set
gbm.pred <- predict(gbm.trainingModel, cleanValidationSet)
rf.pred <- predict(rf.trainingModel, cleanValidationSet)
rpart.pred <- predict(rpart.trainingModel, cleanValidationSet)

# create and print the confusion matrices to compare the predictions
gbm.cfm <- confusionMatrix(gbm.pred, cleanValidationSet$classe)
rf.cfm <- confusionMatrix(rf.pred, cleanValidationSet$classe)
rpart.cfm <- confusionMatrix(rpart.pred, cleanValidationSet$classe)

print("rf:")
print(rf.cfm$table)
print(rf.cfm$overall)
print("gbm:")
print(gbm.cfm$table)
print(gbm.cfm$overall)
print("rpart:")
print(rpart.cfm$table)
print(rpart.cfm$overall)

# predict using the best (random forest) model
test.pred <- predict(rf.trainingModel, testSet)

# print out the result
predResults <- data.frame(testSet$problem_id,test.pred)
print(predResults)
write.table(predResults, file="predictionResults.txt", sep="\t")

