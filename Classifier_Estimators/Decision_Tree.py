import os

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import Function as f
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

DATASET_PATH = 'D:\\Universita\\FlakyTest_Detection\\DataSet\\DatasetGenerale.csv'


if __name__ == "__main__":
    best_params=None
    feature_engineering_pipeline=[]

    dataset_flaky = pd.read_csv(DATASET_PATH)
    # Divido il dataset in train set e test set
    stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratifiedShuffleSplit.split(X=dataset_flaky, y=dataset_flaky['isFlaky']):
        train_set = dataset_flaky.loc[train_index]
        test_set = dataset_flaky.loc[test_index]

    # Creo un nuovo esperimento su mlflow
    experiment_id=mlflow.create_experiment("Decision_Tree")

    ##################################################################################################
    # Verifico le prestazioni del classificatore sul dataset, senza la fase di data pre processing
    ##################################################################################################
    with mlflow.start_run(experiment_id=experiment_id,run_name='Decision Tree without Data PreProcessing'):

        ########################################################################################
        # Separo le lable dai campioni e converto il dataframe in un array numpy
        ########################################################################################
        X_train_set=train_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_train_set=train_set[['idProject','nameProject','testCase','isFlaky']]
        X_train_set_numpy = X_train_set.to_numpy()
        y_train_set_numpy = y_train_set['isFlaky'].to_numpy()

        ########################################################################################
        # Eseguo una stima delle prestazioni del modello sul train set,
        # con la cross validation nidificata (10x3)
        ########################################################################################
        gridSearch_DT=[
            {'criterion':['gini','entropy'],
             'splitter':['best','random'],
             }
        ]

        clf=DecisionTreeClassifier()
        f.nested_cross_validation(estimator=clf,gridSearch_param=gridSearch_DT,cv1=3,cv2=2,X=X_train_set_numpy,y=y_train_set_numpy)


        ########################################################################################
        # Fit del modello
        ########################################################################################
        grid_clf=GridSearchCV(clf,param_grid=gridSearch_DT,cv=2,verbose=True,n_jobs=-1,scoring='f1_micro')
        grid_clf.fit(X=X_train_set,y=y_train_set['isFlaky'])
        clf.set_params(**grid_clf.best_params_)
        best_params=grid_clf.best_params_
        clf.fit(X=X_train_set,y=y_train_set['isFlaky'])
        mlflow.log_params(clf.get_params())

        ########################################################################################
        # Validazione del modello sul test set
        ########################################################################################
        X_test_set=test_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_test_set=test_set[['idProject','nameProject','testCase','isFlaky']]
        y_predict=clf.predict(X=X_test_set)
        f.print_confusion_matrix(y_true=y_test_set['isFlaky'],y_pred=y_predict,estimator_name=clf.__class__.__name__)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path=clf.__class__.__name__+"_NoPreProcessing"
        )
        y_test_set['isFlakyPredict']=y_predict
        y_test_set.to_csv(clf.__class__.__name__+' without Data PreProcessing Prediction.csv')
        mlflow.log_artifact(clf.__class__.__name__+' without Data PreProcessing Prediction.csv',clf.__class__.__name__+' Prediction')
        os.remove(clf.__class__.__name__+' without Data PreProcessing Prediction.csv')
        mlflow.end_run()

    ##################################################################################################
    # Verifico le prestazioni del classificatore sul dataset con SMOTE
    ##################################################################################################
    with mlflow.start_run(experiment_id=experiment_id,run_name='Decision Tree with SMOTE'):

        X_train_set=train_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_train_set=train_set['isFlaky']
        X_train_set=X_train_set.to_numpy()
        y_train_set=y_train_set.to_numpy()

        ####################################################
        # Applico SMOTE al dataset
        ###################################################
        f.print_plot_dataset(plot_title="Dataset Non Bilanciato",X=X_train_set,y=y_train_set)
        smote=SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X_train_set,y_train_set=smote.fit_resample(X=X_train_set,y=y_train_set)
        f.print_plot_dataset(plot_title="Dataset Bilanciato",X=X_train_set,y=y_train_set)

        ####################################################
        # Addestro il modello
        ###################################################
        clf=DecisionTreeClassifier()
        clf.set_params(**best_params)
        f.cross_validation(estimator=clf,cv=10,X=X_train_set,y=y_train_set)

        clf.set_params(**best_params)
        clf.fit(X=X_train_set,y=y_train_set)
        mlflow.log_params(clf.get_params())

        ####################################################
        # Valido il modello
        ###################################################
        X_test_set=test_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_test_set=test_set[['idProject','nameProject','testCase','isFlaky']]
        y_predict=clf.predict(X=X_test_set.to_numpy())
        f.print_confusion_matrix(y_true=y_test_set['isFlaky'],y_pred=y_predict,estimator_name=clf.__class__.__name__)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path=clf.__class__.__name__+"_SMOTE"
        )
        y_test_set['isFlakyPredict']=y_predict
        y_test_set.to_csv(clf.__class__.__name__+' with SMOTE.csv')
        mlflow.log_artifact(clf.__class__.__name__+' with SMOTE.csv',clf.__class__.__name__+' Prediction')
        os.remove(clf.__class__.__name__+' with SMOTE.csv')

        mlflow.end_run()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Inizio la creazione della miglior pipeline di data pre processing
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++