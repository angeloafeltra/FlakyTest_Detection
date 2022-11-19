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


    for project in projects:
        print(project)
        '''
        ###########################################
        # Validazione cross-project
        ###########################################
        experiment=mlflow.get_experiment_by_name("Validazione cross project (Ensamble)")
        if not experiment:
            experiment_id=mlflow.create_experiment("Validazione cross project (Ensamble)")
        else:
            experiment_id=experiment.experiment_id

        #Verifico se è stata eseguita gia una run per il progetto
        experiment=mlflow.get_experiment_by_name("Validazione cross project (Ensamble)") #Ottengo un object experiment
        all_run=mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if all_run.empty:
            run_presente=False
        else:
            run_presente=(all_run['tags.mlflow.runName']==project).any()

        if not run_presente: #Run non presente, quindi eseguo una valutazione cross-project
            with mlflow.start_run(experiment_id=experiment_id,run_name=project):
                train_set=dataset_flaky.loc[dataset_flaky['nameProject'] != project]
                test_set=dataset_flaky.loc[dataset_flaky['nameProject'] == project]
                mlflow.log_param("Train Set Flaky",(train_set['isFlaky'] == 1).sum())
                mlflow.log_param("Train Set Non Flaky",(train_set['isFlaky'] == 0).sum())
                mlflow.log_param("Test Set Flaky",(test_set['isFlaky'] == 1).sum())
                mlflow.log_param("Test Set Non Flaky",(test_set['isFlaky'] == 0).sum())
                flaky_detector=FlakyTestDetector(k_neighbors=5)
                flaky_detector.fit(dataset=train_set)
                acc,pre,rec,f1,confmat,plot=flaky_detector.predict(dataset=test_set)
                mlflow.log_metric("Accuracy",acc)
                mlflow.log_metric("Precision",pre)
                mlflow.log_metric("Recall",rec)
                mlflow.log_metric("F1",f1)
                mlflow.log_figure(plot,"Confusion Matrix.png")
                tn,fp,fn,tp=confmat.ravel()
                mlflow.log_param("True Positive",tp)
                mlflow.end_run()
        '''

        ###########################################
        # Validazione with-in project
        ###########################################
        experiment=mlflow.get_experiment_by_name("Validazione with-in project (Ensamble)")
        if not experiment:
            experiment_id=mlflow.create_experiment("Validazione with-in project (Ensamble)")
        else:
            experiment_id=experiment.experiment_id
        #Verifico se è stata eseguita gia una run per il progetto
        experiment=mlflow.get_experiment_by_name("Validazione with-in project (Ensamble)") #Ottengo un object experiment
        all_run=mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if all_run.empty:
            run_presente=False
        else:
            run_presente=(all_run['tags.mlflow.runName']==project).any()

        if not run_presente:
            test_set=dataset_flaky.loc[dataset_flaky['nameProject'] == project]
            test_set=test_set.reset_index(drop=True)
            num_test_flaky=(test_set['isFlaky'] == 1).sum()
            if num_test_flaky>1:
                with mlflow.start_run(experiment_id=experiment_id,run_name=project):
                    stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    for train_index, test_index in stratifiedShuffleSplit.split(X=test_set, y=test_set['isFlaky']):
                        train_set_within = test_set.loc[train_index]
                        test_set_within = test_set.loc[test_index]
                        mlflow.log_param("Test Flaky",(test_set['isFlaky'] == 1).sum())
                        mlflow.log_param("Test Non Flaky",(test_set['isFlaky'] == 0).sum())
                        #flaky_detector=FlakyTestDetector(k_neighbors=(train_set_within['isFlaky'] == 1).sum()-1)
                        flaky_detector=FlakyTestDetector(k_neighbors=2)
                        try:
                            flaky_detector.fit(dataset=train_set_within)
                            acc,pre,rec,f1,confmat,plot=flaky_detector.predict(dataset=test_set_within)
                            mlflow.log_metric("Accuracy",acc)
                            mlflow.log_metric("Precision",pre)
                            mlflow.log_metric("Recall",rec)
                            mlflow.log_metric("F1",f1)
                            mlflow.log_figure(plot,"Confusion Matrix.png")
                            tn,fp,fn,tp=confmat.ravel()
                            mlflow.log_param("True Positive",tp)
                            mlflow.end_run()
                        except:
                            mlflow.set_tag("Eccezione",sys.exc_info()[1])
                            mlflow.end_run()



