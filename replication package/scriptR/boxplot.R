
  library(MASS)
  library(foreign)
  library(nnet)
  library(car)
  library(tidyr)
  library(carData)
  library(ggplot2)
  library(data.table)
  library(dplyr)
  library(reshape2)
  library(effsize)

  options(scipen=100000)
  
  
  # Reading input data
  idFlakies <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormIDFlakies.csv", sep = ",", header = TRUE, stringsAsFactors=FALSE)
  flakeFlagger <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormFlakeFlagger.csv", sep= ",",header = TRUE, stringsAsFactors = FALSE)
  
  dfidFlakies <- data.frame(idFlakies)
  dfFlakeFlagger <- data.frame(flakeFlagger)
  
  #create two df, one with flaky tests and one with no flaky tests
  df_flaky <- dfidFlakies[which(dfidFlakies$isFlaky == 1),]
  df_no_flaky <- dfidFlakies[which(dfidFlakies$isFlaky == 0),]
  
  mFlakyidFlakies<- df_flaky[,4:28] %>% gather(var,value) %>% mutate(set="Flaky")
  mNoFlakyidFlakies <-df_no_flaky[,4:28] %>% gather(var,value) %>% mutate(set="NoFlaky")
  
  # combining them into one dataframe
  AB <- rbind(mNoFlakyidFlakies,mFlakyidFlakies)
  
  # creating the boxplot
  ggplot(AB, aes(x=value, y=var, fill=set)) +
    geom_boxplot(outlier.shape = NA, alpha=0.7) +
    scale_fill_brewer(palette="Set3") +
    theme_bw()+
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text=element_text(size=14, face="bold"), plot.title = element_text(size=15),
         legend.title = element_text(size = 15), legend.text = element_text(size = 15), legend.position = "bottom")+
    ggtitle("Boxplot Independent Variables iDFlakies dataset")
  
  
  #create two df, one with flaky tests and one with no flaky tests
  df_flakyFF <- dfFlakeFlagger[which(dfFlakeFlagger$isFlaky == 1),]
  df_no_flakyFF <- dfFlakeFlagger[which(dfFlakeFlagger$isFlaky == 0),]
  
  mFlakyFF<- df_flakyFF[,4:28] %>% gather(var,value) %>% mutate(set="Flaky")
  mNoFlakyFF <-df_no_flakyFF[,4:28] %>% gather(var,value) %>% mutate(set="NoFlaky")
  
  # combining them into one dataframe
  CD <- rbind(mNoFlakyFF,mFlakyFF)
  
  # creating the boxplot
  ggplot(CD, aes(x=value, y=var, fill=set)) +
    geom_boxplot(outlier.shape = NA, alpha=0.7) +
    scale_fill_brewer(palette="Set3") +
    theme_bw()+
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text=element_text(size=14, face="bold"), plot.title = element_text(size=15),
          legend.title = element_text(size = 15), legend.text = element_text(size = 15), legend.position = "bottom")+
    ggtitle("Boxplot Independent Variables FlakeFlagger dataset")
  