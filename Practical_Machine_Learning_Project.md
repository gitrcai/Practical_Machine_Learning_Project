Practical Machine Learning Project
----------------------------------

Intruduction:
-------------

This project will use data set from accelerometers on the belt, forearm,
arm, and dumbell of 6 participant to build model on the training data
set and predict outcome on the test data set. I will do some pre-process
on the data sets. Specially, remove some variables which are constant or
missing in most situation. Then, I will divide training set into a new
training set and a validation data set. I will build two different
models and validate the models on the validation data set. finally, I
will pick a better model applying the model to the test data set and
predict the outcome "classe".

Loading and Processing the data
-------------------------------

    ##Load required packages
    library(caret)
    library(rpart)
    library(rpart.plot)
    library(RColorBrewer)
    library(rattle)
    library(randomForest)
    library(gbm)
    library(survival)
    library(plyr)

    ##Load data from web to my working directory
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="./pm1-training.csv")
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="./pm1-testing.csv")
    ##Read data to R
    training <- read.csv("pm1-training.csv")
    testing <- read.csv("pm1-testing.csv")

There are 19622 observations and 160 variables in training data set.
There are 20 observationa and 160 variables in the testing data set. By
viewing training data set, I find out there are some variables with
large number of mising values and also some variables will not have any
impact on our prediction project such as x, user\_name, time variables,
and the almost constant variables. Nest step, I will remove these
variables.

    ##Remove variables near zero variance to reduce variables from 160 to 100
    nzv <- nearZeroVar(training)
    training <- training[, -nzv]
    ##Remove variables with a lot of NA to reduce variables from 100 to 59
    lotsNA <- sapply(training, function(x) mean(is.na(x))) > 0.90
    training <- training[, lotsNA==F]
    ##Remove no impact variables such as ID, name...,to reduce variables from 59 to 54
    training <- training[, -(1:5)]
    ##Do the same thing on the testong data set
    nzvt <- nearZeroVar(testing)
    testing <- testing[, -nzvt]
    lotsNAt <- sapply(testing, function(x) mean(is.na(x))) > 0.90
    testing <- testing[, lotsNAt==F]
    testing <- testing[, -(1:5)]

Now, lets divide training data set into two data sets. one of them to
use for validattion

    set.seed(123)
    inTrain <- createDataPartition(y=training$classe, p=0.7, list=F)
    training <- training[inTrain, ]
    validation <- training[-inTrain, ]
    dim(training)

    ## [1] 13737    54

    dim(testing)

    ## [1] 20 54

Building Model
--------------

    ##Using three different method to build model on training set
    fitControl <- trainControl(method="cv", number=3, verboseIter=F)
    ##Decision trees method (rpart)
    modFit1 <- train(classe ~ ., data=training, method="rpart", trControl=fitControl)
    ##Stochastic gradient boosting trees (gbm)
    modFit2 <- train(classe ~ ., data=training, method="gbm", trControl=fitControl)
    ##Random forest decision trees (rf)
    modFit3 <- train(classe ~. , data=training, method="rf", trControl=fitControl)

    fancyRpartPlot(modFit1$finalModel)

![](Practical_Machine_Learning_Project_files/figure-markdown_strict/unnamed-chunk-6-1.png)

    modFit2

    ## Stochastic Gradient Boosting 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9158, 9158, 9158 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7610832  0.6969436
    ##   1                  100      0.8326418  0.7881849
    ##   1                  150      0.8681663  0.8331399
    ##   2                   50      0.8817063  0.8501876
    ##   2                  100      0.9381233  0.9217079
    ##   2                  150      0.9617820  0.9516456
    ##   3                   50      0.9284414  0.9094362
    ##   3                  100      0.9664410  0.9575399
    ##   3                  150      0.9844216  0.9802936
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

    modFit3$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.2%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    5 2650    3    0    0 0.0030097818
    ## C    0    4 2392    0    0 0.0016694491
    ## D    0    0    9 2243    0 0.0039964476
    ## E    0    0    0    4 2521 0.0015841584

    ##Prediction on validation data set
    validpred1 <- predict(modFit1, validation)
    validpred2 <- predict(modFit2, validation)
    validpred3 <- predict(modFit3, validation)
    confusionMatrix(validpred1, validation$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1004  177   92  150   40
    ##          B   67  440   71  142  127
    ##          C   88  174  576  331  112
    ##          D    0    0    0    0    0
    ##          E    2    0    0   32  492
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6102          
    ##                  95% CI : (0.5951, 0.6251)
    ##     No Information Rate : 0.282           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5005          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8648   0.5563   0.7794   0.0000   0.6381
    ## Specificity            0.8447   0.8776   0.7913   1.0000   0.9898
    ## Pos Pred Value         0.6863   0.5195   0.4496      NaN   0.9354
    ## Neg Pred Value         0.9408   0.8927   0.9425   0.8409   0.9223
    ## Prevalence             0.2820   0.1921   0.1795   0.1591   0.1873
    ## Detection Rate         0.2439   0.1069   0.1399   0.0000   0.1195
    ## Detection Prevalence   0.3554   0.2057   0.3111   0.0000   0.1278
    ## Balanced Accuracy      0.8547   0.7169   0.7854   0.5000   0.8140

    confusionMatrix(validpred2, validation$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1161    7    0    0    0
    ##          B    0  781    3    1    2
    ##          C    0    3  733    7    1
    ##          D    0    0    2  647    4
    ##          E    0    0    1    0  764
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9925          
    ##                  95% CI : (0.9893, 0.9949)
    ##     No Information Rate : 0.282           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9905          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9874   0.9919   0.9878   0.9909
    ## Specificity            0.9976   0.9982   0.9967   0.9983   0.9997
    ## Pos Pred Value         0.9940   0.9924   0.9852   0.9908   0.9987
    ## Neg Pred Value         1.0000   0.9970   0.9982   0.9977   0.9979
    ## Prevalence             0.2820   0.1921   0.1795   0.1591   0.1873
    ## Detection Rate         0.2820   0.1897   0.1780   0.1572   0.1856
    ## Detection Prevalence   0.2837   0.1912   0.1807   0.1586   0.1858
    ## Balanced Accuracy      0.9988   0.9928   0.9943   0.9930   0.9953

    confusionMatrix(validpred3, validation$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1161    0    0    0    0
    ##          B    0  791    0    0    0
    ##          C    0    0  739    0    0
    ##          D    0    0    0  655    0
    ##          E    0    0    0    0  771
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9991, 1)
    ##     No Information Rate : 0.282      
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity             1.000   1.0000   1.0000   1.0000   1.0000
    ## Specificity             1.000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value          1.000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value          1.000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence              0.282   0.1921   0.1795   0.1591   0.1873
    ## Detection Rate          0.282   0.1921   0.1795   0.1591   0.1873
    ## Detection Prevalence    0.282   0.1921   0.1795   0.1591   0.1873
    ## Balanced Accuracy       1.000   1.0000   1.0000   1.0000   1.0000

From my three methods of prediction, The ramdom forest method have high
accruracy. I will apply this model to test data set.

Generate Files to submit
------------------------

    ##predict on test data set
    testpred <- predict(modFit3, newdata=testing)
    testpred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

    # create function to write predictions to files
    pml_write_files <- function(x) {
        n <- length(x)
        for(i in 1:n) {
            filename <- paste0("problem_id_", i, ".txt")
            write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
        }
    }

    # create prediction files to submit
    pml_write_files(as.character(testpred))
