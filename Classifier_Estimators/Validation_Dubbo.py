import sys

from sklearn.model_selection import StratifiedShuffleSplit
from FlakyTestDetector import FlakyTestDetector
import mlflow
import pandas as pd
import numpy as np

DATASET_PATH = 'D:\\Universita\\FlakyTest_Detection\\DataSet\\DatasetGenerale2.csv'

if __name__ == "__main__":
    dataset_flaky = pd.read_csv(DATASET_PATH)
    projects=dataset_flaky['nameProject'].unique() #Ottengo i nomi di tutti i progetti
    project='dubbo'


    ###########################################
    # Validazione cross-projectdubbo
    ###########################################
    experiment=mlflow.get_experiment_by_name("Validazione Dubbo")
    if not experiment:
        experiment_id=mlflow.create_experiment("Validazione Dubbo")
    else:
        experiment_id=experiment.experiment_id

    #Verifico se è stata eseguita gia una run per il progetto
    run_presente=False
    if not experiment_id is None:
        all_run=mlflow.search_runs(experiment_ids=[experiment_id])
        if all_run.empty:
            run_presente=False
        else:
            run_presente=(all_run['tags.mlflow.runName']==project).any()

    if not run_presente: #Run non presente, quindi eseguo una valutazione cross-project
        with mlflow.start_run(experiment_id=experiment_id,run_name=project):
            dataset_whitout_dubbo=dataset_flaky.loc[dataset_flaky['nameProject'] != project]
            dataset_whitout_dubbo=dataset_whitout_dubbo.reset_index(drop=True)
            dataset_whitout_dubbo.to_csv("Dataset_Without_dubbo.csv",index=False)
            dubbo=dataset_flaky.loc[dataset_flaky['nameProject'] == project]
            dubbo=dubbo.reset_index(drop=True)
            dubbo.to_csv("Dubbo.csv",index=False)
            mlflow.log_param("Dataset without dubbo Flaky",(dataset_whitout_dubbo['isFlaky'] == 1).sum())
            mlflow.log_param("Dataset without dubbo Non Flaky",(dataset_whitout_dubbo['isFlaky'] == 0).sum())
            mlflow.log_param("Dubbo Flaky",(dubbo['isFlaky'] == 1).sum())
            mlflow.log_param("Dubbo Non Flaky",(dubbo['isFlaky'] == 0).sum())

            #Divido il dataset in train e test
            stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in stratifiedShuffleSplit.split(X=dataset_whitout_dubbo, y=dataset_whitout_dubbo['isFlaky']):
                train_set = dataset_whitout_dubbo.loc[train_index]
                test_set = dataset_whitout_dubbo.loc[test_index]

            #Addestro
            flaky_detector=FlakyTestDetector(k_neighbors=5)
            flaky_detector.fit(dataset=train_set)
            #Valido
            acc,pre,rec,f1,confmat,plot=flaky_detector.predict(dataset=test_set)
            mlflow.log_metric("Accuracy Dataset",acc)
            mlflow.log_metric("Precision Dataset",pre)
            mlflow.log_metric("Recall Dataset",rec)
            mlflow.log_metric("F1 Dataset",f1)
            mlflow.log_figure(plot,"Confusion Matrix.png")
            tn,fp,fn,tp=confmat.ravel()
            mlflow.log_param("True Positive Dataset",tp)

            #Validation con dubbo
            acc,pre,rec,f1,confmat,plot=flaky_detector.predict(dataset=dubbo)
            mlflow.log_metric("Accuracy Dubbo",acc)
            mlflow.log_metric("Precision Dubbo",pre)
            mlflow.log_metric("Recall Dubbo",rec)
            mlflow.log_metric("F1 Dubbo",f1)
            mlflow.log_figure(plot,"Confusion Matrix.png")
            tn,fp,fn,tp=confmat.ravel()
            mlflow.log_param("True Positive Dubbo",tp)
            mlflow.end_run()


