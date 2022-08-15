rm(list = ls())

require(tsutils)
options(max.print = 10000)


#this is the Nemenyi test for RQ3 for FlakeFlagger dataset
flakeFlagger <-
  read.csv("/your path/replication package/result/resultRQ3/resultFlakeFlaggerNew.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)


undersampleFF <- matrix(c(
  flakeFlagger[flakeFlagger$ModelBalance == "randomForest", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestrandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestrandomunder", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestnearmissunder1", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestnearmissunder2", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestnearmissunder3", ]$f1_score),nrow=length(flakeFlagger[flakeFlagger$ModelBalance == "randomForest", ]$f1_score), ncol=8, 
  dimnames=list(1:length(flakeFlagger[flakeFlagger$ModelBalance == "randomForest", ]$f1_score),c(
    "randomForest",
    "randomForestBorderlineSMOTE",
    "randomForestSMOTE",
    "randomForestrandomover",
    "randomForestrandomunder",
    "randomForestnearmissunder1",
    "randomForestnearmissunder2",
    "randomForestnearmissunder3")))

nemenyi(undersampleFF, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood F-Measure FlakeFlagger Random Forest",xlab="",title=NULL,main=NULL,sub=NULL, cex.axis=3,cex.lab=1,cex.main=3,cex.sub=2)


if(F) {

y <- matrix(c(
  flakeFlagger[flakeFlagger$ModelBalance == "adaboost", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "adaboostBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "adaboostSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "adaboostrandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "decisionTree", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "decisionTreeBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "decisionTreeSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "decisionTreerandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "MLP", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "MLPBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "MLPSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "MLPrandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "naiveBayes", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "naiveBayesBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "naiveBayesSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "naiveBayesrandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForest", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "randomForestrandomover", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "SVM", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "SVMBorderlineSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "SVMSMOTE", ]$f1_score,
  flakeFlagger[flakeFlagger$ModelBalance == "SVMrandomover", ]$f1_score),nrow=length(flakeFlagger[flakeFlagger$ModelBalance == "adaboost", ]$f1_score), ncol=24, 
  dimnames=list(1:length(flakeFlagger[flakeFlagger$ModelBalance == "adaboost", ]$f1_score),c("adaboost",
                                                                                    "adaboostBorderlineSMOTE",
                                                                                    "adaboostSMOTE",
                                                                                    "adaboostrandomover",
                                                                                    "decisionTree",
                                                                                    "decisionTreeBorderlineSMOTE",
                                                                                    "decisionTreeSMOTE",
                                                                                    "decisionTreerandomover",
                                                                                    "MLP",
                                                                                    "MLPBorderlineSMOTE",
                                                                                    "MLPSMOTE",
                                                                                    "MLPrandomover",
                                                                                    "naiveBayes",
                                                                                    "naiveBayesBorderlineSMOTE",
                                                                                    "naiveBayesSMOTE",
                                                                                    "naiveBayesrandomover",
                                                                                    "randomForest",
                                                                                    "randomForestBorderlineSMOTE",
                                                                                    "randomForestSMOTE",
                                                                                    "randomForestrandomover",
                                                                                    "SVM",
                                                                                    "SVMBorderlineSMOTE",
                                                                                    "SVMSMOTE",
                                                                                    "SVMrandomover")))
                                      
nemenyi(y, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood F-Measure FlakeFlagger",xlab="",title=NULL,main=NULL,sub=NULL, cex.axis=3,cex.lab=1,cex.main=3,cex.sub=2)
}


#this is the Nemenyi test for RQ3 for iDFlakies dataset
dataidFlakies <-
  read.csv("/your path/replication package/result/resultRQ3/resultiDFlakiesNew.csv",
           header = TRUE,
           sep = ",",
           dec = ".")
dataidFlakies <- data.frame(dataidFlakies)

undersampleIF <- matrix(c(
  dataidFlakies[dataidFlakies$ModelBalance == "randomForest", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestrandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestrandomunder", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestnearmissunder1", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestnearmissunder2", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestnearmissunder3", ]$f1_score),nrow=length(dataidFlakies[dataidFlakies$ModelBalance == "randomForest", ]$f1_score), ncol=8, 
  dimnames=list(1:length(dataidFlakies[dataidFlakies$ModelBalance == "randomForest", ]$f1_score),c("randomForest",
                                                                                                   "randomForestBorderlineSMOTE",
                                                                                                   "randomForestSMOTE",
                                                                                                   "randomForestrandomover",
                                                                                                   "randomForestrandomunder",
                                                                                                   "randomForestnearmissunder1",
                                                                                                   "randomForestnearmissunder2",
                                                                                                   "randomForestnearmissunder3"
                                                                                                   )))
   

nemenyi(undersampleIF, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood F-Measure iDFlakies Random Forest",xlab="",title=NULL,main=NULL,sub=NULL, cex.axis=3,cex.lab=1,cex.main=3,cex.sub=2)

z <- matrix(c(
  dataidFlakies[dataidFlakies$ModelBalance == "adaboost", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "adaboostBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "adaboostSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "adaboostrandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "decisionTree", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "decisionTreeBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "decisionTreeSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "decisionTreerandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "MLP", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "MLPBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "MLPSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "MLPrandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "naiveBayes", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "naiveBayesBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "naiveBayesSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "naiveBayesrandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForest", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "randomForestrandomover", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "SVM", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "SVMBorderlineSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "SVMSMOTE", ]$f1_score,
  dataidFlakies[dataidFlakies$ModelBalance == "SVMrandomover", ]$f1_score),nrow=length(dataidFlakies[dataidFlakies$ModelBalance == "adaboost", ]$f1_score), ncol=24, 
  dimnames=list(1:length(dataidFlakies[dataidFlakies$ModelBalance == "adaboost", ]$f1_score),c("adaboost",
                                                                                           "adaboostBorderlineSMOTE",
                                                                                           "adaboostSMOTE",
                                                                                           "adaboostrandomover",
                                                                                           "decisionTree",
                                                                                           "decisionTreeBorderlineSMOTE",
                                                                                           "decisionTreeSMOTE",
                                                                                           "decisionTreerandomover",
                                                                                           "MLP",
                                                                                           "MLPBorderlineSMOTE",
                                                                                           "MLPSMOTE",
                                                                                           "MLPrandomover",
                                                                                           "naiveBayes",
                                                                                           "naiveBayesBorderlineSMOTE",
                                                                                           "naiveBayesSMOTE",
                                                                                           "naiveBayesrandomover",
                                                                                           "randomForest",
                                                                                           "randomForestBorderlineSMOTE",
                                                                                           "randomForestSMOTE",
                                                                                           "randomForestrandomover",
                                                                                           "SVM",
                                                                                           "SVMBorderlineSMOTE",
                                                                                           "SVMSMOTE",
                                                                                           "SVMrandomover")))

nemenyi(z, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood F-Measure iDFlakies",xlab="",title=NULL,main=NULL,sub=NULL, cex.axis=3,cex.lab=1,cex.main=3,cex.sub=2)





#this is the Nemenyi test for RQ4
dataRQ4 <-
  read.csv("/your path/replication package/result/resultRQ4/resultAllProjects.csv",
           header = TRUE,
           sep = ",",
           dec = ".")



x <- matrix(c(
  dataRQ4[dataRQ4$features_structure == "Flake-Flagger-Features", ]$FM,
  dataRQ4[dataRQ4$features_structure == "vocabulary-Features", ]$FM,
  dataRQ4[dataRQ4$features_structure == "vocabulary-with-flakeFlagger", ]$FM,
  dataRQ4[dataRQ4$features_structure == "myapproach", ]$FM
),nrow=length(dataRQ4[dataRQ4$features_structure == "Flake-Flagger-Features", ]$FM), ncol=4, 
dimnames=list(1:length(dataRQ4[dataRQ4$features_structure == "Flake-Flagger-Features", ]$FM),c("FlakeFlagger",
                                                                              "Vocabulary",
                                                                              "Combined",
                                                                              "Static Approach"
                                                                             
)))
nemenyi(x, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood F-Measure for RQ4",xlab="",title=NULL,main=NULL,sub=NULL, cex.axis=3,cex.lab=1,cex.main=3,cex.sub=2)

