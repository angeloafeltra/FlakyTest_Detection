# library
library(ggplot2)

# create a dataset
firstNineProjects <- read.csv("/your path/replication package/result/resultRQ4/resultFirstNineProjects.csv",
                              header = TRUE,
                              sep = ",",
                              dec = ".")
firstNineProjects <- data.frame(firstNineProjects)

# Grouped
ggplot(firstNineProjects, aes(fill=features_structure, y=FM, x=project)) + 
  geom_bar(position="dodge", stat="identity") +
  facet_wrap(~project) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        strip.background = element_blank())


# create a dataset
lastNineProjects <- read.csv("/your path/replication package/result/resultRQ4/resultLastNineProjects.csv",
                              header = TRUE,
                              sep = ",",
                              dec = ".")
lastNineProjects <- data.frame(lastNineProjects)

# Grouped
ggplot(lastNineProjects, aes(fill=features_structure, y=FM, x=project)) + 
  geom_bar(position="dodge", stat="identity") +
  facet_wrap(~project) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        strip.background = element_blank())

