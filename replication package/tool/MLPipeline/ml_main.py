from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from ml_utils import get_params, id_string, out_dir
from ml_preparation import data_cleaning, feature_selection, data_balancing, column_drop, vexctorizeToken
from ml_classification import get_clf, hyperparam_opt, scorer, save_best_params
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
import pickle
from halo import Halo
import json
import gc
import os
import numpy as np

print("________________________________")
print("Starting")
# get pipeline customization parameters and setup output directories
params = get_params()
k = params["k"]
#k = 10
start_time = datetime.now()
time_str = start_time.strftime("%Y%m%d_%H%M%S")
job_id = id_string(params)
out_dir = out_dir(job_id + "__" + time_str)
perf_df = pd.DataFrame([])
fpr_list = []
tpr_list = []

print("Started: " + start_time.strftime("%d %m %Y H%H:%M:%S"))
print("Results will be saved to dir: " + out_dir)
# get dataset
dataset_dir = "your path/replication package/dataset/datasetRQ3/flakeFlagger/" + params[
    "data"] + ".csv"  # Insert here your path

#this line is for RQ3    
df = pd.read_csv(dataset_dir)

#LINES from 43 to 51 are for RQ4
#the next four lines are for tokenize the token in the dataset 
#vocabulary_processed_data = pd.read_csv(dataset_dir)
#tokenOnly = vexctorizeToken(vocabulary_processed_data['tokenList'])  
#vocabulary_processed_data = vocabulary_processed_data.drop(columns=['tokenList'])
#df = pd.concat([vocabulary_processed_data, tokenOnly.reindex(vocabulary_processed_data.index)], axis=1)
#vectorize = CountVectorizer()
#train = vectorize.fit_transform(df)
#matrix_token = pd.DataFrame(train.toarray(), columns=vectorize.get_feature_names())
#matrix_token2 = pd.concat([df, matrix_token.reindex(df.index)], axis=1)

print("Loaded dataset of size: ")
print(df.shape)
print("Splitting dataset...")
# split dataset in k folds
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#kfold = KFold(n_splits=k, shuffle=True, random_state=42)

X = df.iloc[: , [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]].copy() #this line is for RQ3
#X = df.iloc [: , [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].copy() #this line is for create the comparison with flakeflagger
X = df.iloc[:, df.columns != 'isFlaky'].copy()
#y = df.iloc[: , [29]].copy()
y = df["isFlaky"]
folds = kfold.split(X,y)

# del df
gc.collect()
i = 0
cols = df.select_dtypes([np.int64, np.int32]).columns
df[cols] = np.array(df[cols], dtype=float)
#df[cols] = df[cols].astype(float)

print_header = True

for fold in folds:
    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Get data fold")


    # get train and test features and labels
    train = df.iloc[fold[0]]
    test = df.iloc[fold[1]]



    y_train = train["isFlaky"]
    y_train = y_train.astype(int)
    #this line is for RQ3
    X_train = train.drop(columns=["id","nameProject","testCase","isFlaky"])
    testset_sample = test[["id", "projectName","test_name","isFlaky"]]
    
    #lines 92-92 are for RQ4
    #X_train = train.drop(columns=["id","projectName", "test_name", "isFlaky", "java_keywords"])
    #testset_sample = test[["id", "projectName","test_name","isFlaky", "java_keywords"]]
    
    y_test = test["isFlaky"]
    y_test = y_test.astype(int)
    
    X_test = test.drop(columns = ["id", "projectName", "testCase", "isFlaky"])
    #line 99 is for RQ4
    #X_test = test.drop(columns=["id", "projectName", "test_name", "isFlaky", "java_keywords"])
    
    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Data cleaning")
  

    columns_to_retain = X_train.columns
    X_test = X_test[columns_to_retain]
    if not params["feature_sel"] == "none":
        print("Round " + str(i + 1) + " of " + str(k) + ": " + "Feature selection")
        columns_to_retain = feature_selection(params["feature_sel"], X_train)
        X_train = X_train[columns_to_retain]
        X_test = X_test[columns_to_retain]



    data = []
    data = [columns_to_retain, mutual_info_classif(X_train[columns_to_retain], y_train, discrete_features = True)]


    data_T = pd.DataFrame(data).T
    data_T.columns = ["variable", "value"]

    data_filter = data_T[data_T.value >= 0.001]
    X_train = X_train[data_filter.variable]
    X_test = X_test[data_filter.variable]

    with open (out_dir + "IG/" + str(i) + ".txt", 'w') as f:
        dfAsString = data_T.to_string(header=False, index=False)
        f.write(dfAsString)
    del columns_to_retain
    gc.collect()


    # fix bug with numpy arrays
    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_test = X_test.values
    y_test = y_test.values.ravel()

    # data balancing
    if not params["balancing"] == "none":
        print("Round " + str(i + 1) + " of " + str(k) + ": " + "Data balancing")
        X_train, y_train = data_balancing(params["balancing"], X_train, y_train)

    # classifier 
    if not params["classifier"] == "none":
        clf_name = params["classifier"]
    else:
        clf_name = "dummy_random"
    clf = get_clf(clf_name)

    # hyperparameter opt
    if not params["optimization"] == "none" and not clf_name.startswith("dummy"):
        print("Round " + str(i + 1) + " of " + str(k) + ": " + "Hyperparameters optimization")
        best_params = hyperparam_opt(clf, clf_name, params["optimization"], X_train, y_train)
        save_best_params(best_params, out_dir + "best_params/" + str(i))
        clf.set_params(**best_params)

    # validation 
    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Training")
    clf.fit(X_train, y_train)


    del X_train
    del y_train
    gc.collect()

    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Testing")
    fpr, tpr, res, y_pred = scorer(clf, clf_name, X_test, y_test)
    y_pred = pd.DataFrame(y_pred, columns =["prediction"], index=testset_sample.index)
    y_pred.replace({0.0:False,1.0:True}, inplace=True)

    agreement = testset_sample.join(y_pred)
    mode = 'w' if print_header else 'a'
    agreement.to_csv(out_dir+"resultForTestCase.csv", mode=mode, header=print_header)
    print_header =False
    pyplot.figure()
    pyplot.plot(fpr, tpr)
    pyplot.savefig(out_dir + "roc_curves/" + str(i) + ".png")
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    perf_df = pd.concat([perf_df, res], ignore_index=True)
 

    del X_test
    del y_test
    gc.collect()

    i = i + 1


print("Saving performance")

sumTN = perf_df['tn'].sum()
sumFP = perf_df['fp'].sum()
sumFN = perf_df['fn'].sum()
sumTP = perf_df['tp'].sum()
meanPR = perf_df['precision'].mean()
meanRC = perf_df['recall'].mean()
meanACC = perf_df['accuracy'].mean()
meanIR = perf_df['inspection_rate'].mean()
meanF1 = perf_df['f1_score'].mean()
meanMCC = perf_df['mcc'].mean()
meanAUC = perf_df['auc_roc_score'].mean()

list = [sumTN, sumFP, sumFN, sumTP,meanPR,meanRC,meanACC, meanIR, meanF1, meanMCC, meanAUC]
#perf_df  = pd.read_csv('performance.csv')
perf_df = perf_df.append(pd.Series(list, index=perf_df.columns[:len(list)]), ignore_index=True)
perf_df.to_csv(out_dir + "performance.csv")


pyplot.figure()
for i in range(k):
    pyplot.plot(fpr_list[i], tpr_list[i])
pyplot.savefig(out_dir + "roc_curves/all.png")

end_time = datetime.now()
print(params)
print("Ended: " + end_time.strftime("%d %m %Y H%H:%M:%S"))
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
with open(out_dir + "elapsed_time.json", 'w') as f:
    f.write(json.dumps({"elapsed_time": f"{elapsed_time}"}))

print("________________________________")

