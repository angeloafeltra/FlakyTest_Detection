library(MASS)
library(foreign)
library(nnet)
library(stargazer)
library(car)
library(tidyr)
library(effsize)
library(nortest)

options(scipen=100000)

# Reading input data
idFlakies <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormIDFlakies.csv", sep = ",", header = TRUE, stringsAsFactors=FALSE)
flakeFlagger <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormFlakeFlagger.csv", sep = ",", header = TRUE, stringsAsFactors=FALSE)

#See the types of my data
str(idFlakies)
dfidFlakies <- data.frame(idFlakies)

str(flakeFlagger)
dfidFlakies <- data.frame(flakeFlagger)

# Applying a Logistic Regression model for iDFlakies dataset
# Applying a Logistic Regression model using the binomial function
model <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
              + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$loc + idFlakies$lcom2
              + idFlakies$lcom5 + idFlakies$cbo + idFlakies$wmc + idFlakies$rfc + idFlakies$mpc + idFlakies$halsteadVocabulary + idFlakies$halsteadLength + idFlakies$halsteadVolume   
              + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
             family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model))


# Applying a Logistic Regression model using the binomial function without RFC
model2 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$loc + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo + idFlakies$wmc + idFlakies$rfc + idFlakies$mpc + idFlakies$halsteadVocabulary + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model2))


# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model3 <-  glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
                + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$loc + idFlakies$lcom2
                + idFlakies$lcom5 + idFlakies$cbo + idFlakies$wmc  + idFlakies$mpc + idFlakies$halsteadVocabulary + idFlakies$halsteadLength    
                + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
               family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model3))


# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model4 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo + idFlakies$wmc  + idFlakies$mpc + idFlakies$halsteadVocabulary + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model4))


# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model5 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo + idFlakies$mpc + idFlakies$halsteadVocabulary + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model5))


# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model6 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo + idFlakies$mpc + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))
#Applying the VIF function to verify the multi-collinearity
#print(vif(model6))


# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model7 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo  + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate + idFlakies$complexClass + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model7))

# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model8 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo  + idFlakies$halsteadLength    
               + idFlakies$classDataShouldBePrivate  + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model8))

# Applying a Logistic Regression model using the binomial function without RFC and Halstead Volume
model9 <- glm((idFlakies$isFlaky ~ idFlakies$tloc + idFlakies$tmcCabe  + idFlakies$assertionDensity  + idFlakies$assertionRoulette + idFlakies$mysteryGuest + idFlakies$eagerTest
               + idFlakies$sensitiveEquality  + idFlakies$resourceOptimism + idFlakies$conditionalTestLogic + idFlakies$fireAndForget + idFlakies$lcom2
               + idFlakies$lcom5 + idFlakies$cbo + idFlakies$classDataShouldBePrivate  + idFlakies$functionalDecomposition + idFlakies$godClass + idFlakies$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model9))


# Save in a txt file
sink("/your path/result/result_RQ1_RQ2/DatasetIDFlakies/output_GLM.txt")
print(summary(model9))
sink()

# Applying a Logistic Regression model for FlakeFlagger dataset
# Applying a Logistic Regression model using the binomial function
model <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
              + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget + flakeFlagger$loc + flakeFlagger$lcom2
              + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$wmc + flakeFlagger$rfc + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength + flakeFlagger$halsteadVolume   
              + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass + flakeFlagger$spaghettiCode), 
             family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model))

# Applying a Logistic Regression model using the binomial function
model2 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
              + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget + flakeFlagger$loc + flakeFlagger$lcom2
              + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$wmc + flakeFlagger$rfc + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength   
              + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass + flakeFlagger$spaghettiCode), 
             family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model2))

# Applying a Logistic Regression model using the binomial function
model3 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget + flakeFlagger$loc + flakeFlagger$lcom2
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$wmc  + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass + flakeFlagger$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model3))

# Applying a Logistic Regression model using the binomial function
model4 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget  + flakeFlagger$lcom2
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$wmc  + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass + flakeFlagger$spaghettiCode), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model4))

# Applying a Logistic Regression model using the binomial function
model5 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget  + flakeFlagger$lcom2
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$wmc  + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model5))

# Applying a Logistic Regression model using the binomial function
model6 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget  + flakeFlagger$lcom2
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary + flakeFlagger$halsteadLength   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model6))

# Applying a Logistic Regression model using the binomial function
model7 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget  + flakeFlagger$lcom2
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model7))

# Applying a Logistic Regression model using the binomial function
model8 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget 
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$mpc + flakeFlagger$halsteadVocabulary   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model8))

# Applying a Logistic Regression model using the binomial function
model9 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget 
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$halsteadVocabulary   
               + flakeFlagger$classDataShouldBePrivate + flakeFlagger$complexClass + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model9))

# Applying a Logistic Regression model using the binomial function
model10 <- glm((flakeFlagger$isFlaky ~ flakeFlagger$tloc + flakeFlagger$tmcCabe  + flakeFlagger$assertionDensity  + flakeFlagger$assertionRoulette + flakeFlagger$mysteryGuest + flakeFlagger$eagerTest
               + flakeFlagger$sensitiveEquality  + flakeFlagger$resourceOptimism + flakeFlagger$conditionalTestLogic + flakeFlagger$fireAndForget 
               + flakeFlagger$lcom5 + flakeFlagger$cbo + flakeFlagger$halsteadVocabulary   
               + flakeFlagger$classDataShouldBePrivate  + flakeFlagger$functionalDecomposition + flakeFlagger$godClass), 
              family=binomial(link="logit"))

#Applying the VIF function to verify the multi-collinearity
#print(vif(model10))

# Save in a txt file
sink("/your path/result/result_RQ1_RQ2/DatasetFlakeFlagger/output_GLM.txt")
print(summary(model10))
sink()
