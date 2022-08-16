#README

REPLICATION PACKAGE FOLDER EXPLANATION

This folder contains three folders:
1)dataset:This folder contains the CSV files with all the factors computed for each test case. There is information about the name of test case, the value of each factors and information about the flakiness; the data are already normalized.
Moreover, the datasets are grouped into subfolders based on the RQ in which they are used. There is also a CSV file called "list of projects discarded" in which are reported the projects not considered in our study.
2)result: This folder contains a list of subfolders in whiche are reported the results for each RQ and for the Nemenyi test applied in RQ3 and RQ4.
3)script: This folder contains the scripts (.R) used to build the boxplots, the logistic regression model, Mann Whitney and Cliff's Delta tests and the Nemenyi test.
4)tool: This folder contains the tool used to create our dataset and our machine learning pipeline.

-------------------------------------------------------------------------------------------------------

STEP TO REPRODUCE THE STUDY FOR RQ1 AND RQ2

1) Follow the file "install.txt" to download R Studio
2) Once R Studio is installed, open the application
3) From there, open the file "logisticRegressionModel.R". This file is contained in the folder Script_R. The first lines of this file contain the instructions that will install the packages required to run the analysis.

The other files just will "call" the libraries, but without installing them again. The rest of file contains the instruction to run the statistical analysis.

The file "boxplot.R" contains the instructions for creating the boxplots for each factor.

The file "mannWhitneyAndCliffDelta.R" contains the instructions for running the Mann Whitney and Cliff's Delta tests for each factor.

The file "nemenyi_FMeasure.R" contains the instructions for running the Nemenyi test to evaluate the F-Measure of each Machine Learning model analyzed in RQ3 and RQ4.

4)IMPORTANT: set correctly your path in each script. In particular, there are five instructions that need to be changed:

	1) idFlakies <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormIDFlakies.csv", sep = ",", header = TRUE, stringsAsFactors=FALSE)
  	2) flakeFlagger <- read.csv("/your path/dataset/dataset_RQ1_RQ2/datasetNormFlakeFlagger.csv", sep= ",",header = TRUE, stringsAsFactors = FALSE)
	3) sink("/your path/result/result_RQ1_RQ2/DatasetIDFlakies/output_GLM.txt")
	4) sink("/your path/result/result_RQ1_RQ2/DatasetFlakeFlagger/output_GLM.txt")
	5) sink("/your path/result/result_RQ1_RQ2/DatasetFlakeFlagger/output_Mann-Whitney.txt")
	6) sink("/your path/result/result_RQ1_RQ2/DatasetIDFlakies/output_cliffDelta.txt")

	The first two instructions will have to be changed in every script in the script folder.

	You should change the the first path of the path. The first instruction help to find and load the data to run the analysis, the second will be the path where R will save the output while the third will be the path where R will save the data after the normalization.

5) Select all the instruction of the file and press the button run. When the process is completed, you should find the output written in a text file in the folder "result".

-------------------------------------------------------------------------------------------------------
STEP TO REPRODUCE THE STUDY FOR RQ3 AND RQ4

1) Follow the file "install.txt" to download PyCharm.
2) Once R Studio is installed, open the application.
3) From there, open the project "MLPipeline". This file is contained in the folder "tool". 
4) In ml_main, set your path correctly. Moreover, there are some lines commented useful to run again the experiments related to RQ4. If you want to run this, please consider commenting the lines referred to RQ3.
5) In run_configuration, set the data to analyze if you want to apply the vif function and the other parameters related to the number of folds.

-------------------------------------------------------------------------------------------------------
STEP TO CALCULATE THE METRICS

1) Download the project following the link in "infoIDFlakiesProjects.txt" or "infoFlakeFlaggerProjects.txt". Once you have downloaded the project run "git checkout" following the SHA reported in the txt file.
2) Follow the file "install.txt" to download Java 8 and IntelliJ IDEA.
3) From there, open the project "metricsDetector". This project is in the folder "tool".
4) In "RunTestSmellDetection.java" set your paths correctly. Then, run the detector.
5) For the info related to test flakiness, please refers to "https://sites.google.com/view/flakytestdataset" and "https://zenodo.org/record/5014076#.YpTIQWBBz9E".
