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
flakeFlagger <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormFlakeFlagger.csv", sep= ",",header = TRUE, stringsAsFactors = FALSE)

dfidFlakies <- data.frame(idFlakies)
dfFlakeFlagger <- data.frame(flakeFlagger)

# Save in a txt file the results of Mann Whitney test for Flakeflagger dataset
sink("/your path/result/result_RQ1_RQ2/DatasetFlakeFlagger/output_Mann-Whitney.txt")

print(wilcox.test(tloc ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(tmcCabe ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(assertionDensity ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(assertionRoulette ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(mysteryGuest ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(eagerTest ~ isFlaky, data = dfFlakeFlagger))
print(wilcox.test(sensitiveEquality ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(resourceOptimism  ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(conditionalTestLogic  ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(fireAndForget ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(loc ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(lcom2 ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(lcom5 ~ isFlaky,data = dfFlakeFlagger))
print(wilcox.test(cbo ~ isFlaky,data=dfFlakeFlagger))
print(wilcox.test(wmc ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(rfc ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(mpc  ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(halsteadVocabulary ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(halsteadLength ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(halsteadVolume ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(classDataShouldBePrivate ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(complexClass ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(functionalDecomposition ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(godClass ~ isFlaky, data=dfFlakeFlagger))
print(wilcox.test(spaghettiCode ~ isFlaky, data=dfFlakeFlagger))
sink()

# Save in a txt file the results of Cliff Delta's test
sink("/your path/result/result_RQ1_RQ2/DatasetFlakeFlagger/output_cliffDelta.txt")
print("tloc")
print(cliff.delta(tloc ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))
print("tmcCabe")
print(cliff.delta(tmcCabe ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("assertionDensity")
print(cliff.delta(assertionDensity ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("assertionRoulette")
print(cliff.delta(assertionRoulette ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("mysteryGuest")
print(cliff.delta(mysteryGuest ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("eagerTest")
print(cliff.delta(eagerTest ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("sensitiveEquality")
print(cliff.delta(sensitiveEquality ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("resourceOptimism")
print(cliff.delta(resourceOptimism ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("conditionalTestLogic")
print(cliff.delta(conditionalTestLogic ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("fireAndForget")
print(cliff.delta(fireAndForget ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("loc")
print(cliff.delta(loc ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("lcom2")
print(cliff.delta(lcom2 ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("lcom5")
print(cliff.delta(lcom5 ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("cbo")
print(cliff.delta(cbo ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("wmc")
print(cliff.delta(wmc ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("rfc")
print(cliff.delta(rfc ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("mpc")
print(cliff.delta(mpc ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadVocabulary")
print(cliff.delta(halsteadVocabulary ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadLength")
print(cliff.delta(halsteadLength ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadVolume")
print(cliff.delta(halsteadVolume ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("classDataShouldBePrivate")
print(cliff.delta(classDataShouldBePrivate ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("complexClass")
print(cliff.delta(complexClass ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("functionalDecomposition")
print(cliff.delta(functionalDecomposition ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("godClass")
print(cliff.delta(godClass ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("spaghettiCode")
print(cliff.delta(spaghettiCode ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))
sink()


# Save in a txt file the results of Mann Whitney test for iDFlakies dataset
sink("/your path/result/result_RQ1_RQ2/DatasetIDFlakies/output_Mann-Whitney.txt")

print(wilcox.test(tloc ~ isFlaky, data=dfidFlakies))
print(wilcox.test(tmcCabe ~ isFlaky, data=dfidFlakies))
print(wilcox.test(assertionDensity ~ isFlaky, data=dfidFlakies))
print(wilcox.test(assertionRoulette ~ isFlaky, data=dfidFlakies))
print(wilcox.test(mysteryGuest ~ isFlaky, data=dfidFlakies))
print(wilcox.test(eagerTest ~ isFlaky, data = dfidFlakies))
print(wilcox.test(sensitiveEquality ~ isFlaky, data=dfidFlakies))
print(wilcox.test(resourceOptimism  ~ isFlaky, data=dfidFlakies))
print(wilcox.test(conditionalTestLogic  ~ isFlaky, data=dfidFlakies))
print(wilcox.test(fireAndForget ~ isFlaky, data=dfidFlakies))
print(wilcox.test(loc ~ isFlaky, data=dfidFlakies))
print(wilcox.test(lcom2 ~ isFlaky, data=dfidFlakies))
print(wilcox.test(lcom5 ~ isFlaky,data = dfidFlakies))
print(wilcox.test(cbo ~ isFlaky,data=dfidFlakies))
print(wilcox.test(wmc ~ isFlaky, data=dfidFlakies))
print(wilcox.test(rfc ~ isFlaky, data=dfidFlakies))
print(wilcox.test(mpc  ~ isFlaky, data=dfidFlakies))
print(wilcox.test(halsteadVocabulary ~ isFlaky, data=dfidFlakies))
print(wilcox.test(halsteadLength ~ isFlaky, data=dfidFlakies))
print(wilcox.test(halsteadVolume ~ isFlaky, data=dfidFlakies))
print(wilcox.test(classDataShouldBePrivate ~ isFlaky, data=dfidFlakies))
print(wilcox.test(complexClass ~ isFlaky, data=dfidFlakies))
print(wilcox.test(functionalDecomposition ~ isFlaky, data=dfidFlakies))
print(wilcox.test(godClass ~ isFlaky, data=dfidFlakies))
print(wilcox.test(spaghettiCode ~ isFlaky, data=dfidFlakies))
sink()

# Save in a txt file the results of Cliff Delta's test
sink("/your path/result/result_RQ1_RQ2/DatasetIDFlakies/output_cliffDelta.txt")
print("tloc")
print(cliff.delta(tloc ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))
print("tmcCabe")
print(cliff.delta(tmcCabe ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("assertionDensity")
print(cliff.delta(assertionDensity ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("assertionRoulette")
print(cliff.delta(assertionRoulette ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("mysteryGuest")
print(cliff.delta(mysteryGuest ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("eagerTest")
print(cliff.delta(eagerTest ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("sensitiveEquality")
print(cliff.delta(sensitiveEquality ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("resourceOptimism")
print(cliff.delta(resourceOptimism ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("conditionalTestLogic")
print(cliff.delta(conditionalTestLogic ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("fireAndForget")
print(cliff.delta(fireAndForget ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("loc")
print(cliff.delta(loc ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("lcom2")
print(cliff.delta(lcom2 ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("lcom5")
print(cliff.delta(lcom5 ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("cbo")
print(cliff.delta(cbo ~ isFlaky, data=dfFlakeFlagger ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("wmc")
print(cliff.delta(wmc ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("rfc")
print(cliff.delta(rfc ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("mpc")
print(cliff.delta(mpc ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadVocabulary")
print(cliff.delta(halsteadVocabulary ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadLength")
print(cliff.delta(halsteadLength ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("halsteadVolume")
print(cliff.delta(halsteadVolume ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("classDataShouldBePrivate")
print(cliff.delta(classDataShouldBePrivate ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("complexClass")
print(cliff.delta(complexClass ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("functionalDecomposition")
print(cliff.delta(functionalDecomposition ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("godClass")
print(cliff.delta(godClass ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))

print("spaghettiCode")
print(cliff.delta(spaghettiCode ~ isFlaky, data=dfidFlakies ,conf.level=.95, 
                  use.unbiased=TRUE, use.normal=FALSE, 
                  return.dm=FALSE))
sink()