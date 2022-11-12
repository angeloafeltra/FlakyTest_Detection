import copy
import mlflow
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from Pipeline_Experiment import Pipeline_Experiment
import pandas as pd

DATASET_PATH = 'D:\\Universita\\FlakyTest_Detection\\DataSet\\DatasetGenerale2.csv'


if __name__ == "__main__":

    nome_classificatore='KNN'
    best_params=None
    list_evaluated_method=[]
    list_preProcessing_Pipeline=[]


    dataset_flaky = pd.read_csv(DATASET_PATH)
    # Divido il dataset in train set e test set
    stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratifiedShuffleSplit.split(X=dataset_flaky, y=dataset_flaky['isFlaky']):
        train_set = dataset_flaky.loc[train_index]
        test_set = dataset_flaky.loc[test_index]
    #Separo le lable dei campioni
    X_train_set=train_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
    y_train_set=train_set['isFlaky']
    X_test_set=test_set.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
    y_test_set=test_set[['idProject','nameProject','testCase','isFlaky']]
    print(X_train_set.info())

    # Creo un nuovo esperimento su mlflow se non esiste
    experiment=mlflow.get_experiment_by_name(nome_classificatore)
    if not experiment:
        experiment_id=mlflow.create_experiment(nome_classificatore)
    else:
        experiment_id=experiment.experiment_id


    ##############################################################
    # Pipeline 1 senza data pre processing
    ##############################################################
    print('Pipeline 1')
    parametri_classificatore=[
        {
            'n_neighbors': [1, 10, 20, 30],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    ]

    list_evaluated_method.append(('nested_cross_validation',{
        'gridSearch_param' : parametri_classificatore,
        'cv1':3,
        'cv2':2
    }))

    list_evaluated_method.append(('gridSearch',{
        'gridSearch_param' : parametri_classificatore,
        'cv':2,
    }))

    clf=KNeighborsClassifier()

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=None, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline senza pre processing',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    #############################################################
    # Pipeline 2 con Feature Scaling (Normalization)
    #############################################################
    print('Pipeline 2')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalization',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    #############################################################
    # Pipeline 3 con Feature Scaling (Standardization)
    #############################################################
    print('Pipeline 3')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzation',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 4 con PCA
    ##############################################################
    print('Pipeline 4')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con PCA',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    #############################################################
    # Pipeline 5 con Information Gain
    #############################################################
    print('Pipeline 5')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    accuracy,precision,recall,f1_score_norm=pipeline.run_experiment(mlflow_experiment=experiment_id,
                                                                    mlflow_run_name='Pipeline con Information Gain',
                                                                    X_train_set=copy.copy(X_train_set),
                                                                    X_test_set=copy.copy(X_test_set),
                                                                    y_test_set=copy.copy(y_test_set),
                                                                    y_train_set=copy.copy(y_train_set)
                                                                    )

    ##############################################################
    # Pipeline 6 con SMOTE
    ##############################################################
    print('Pipeline 6')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 7 con Normalizzazione e PCA
    ##############################################################
    print('Pipeline 7')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e PCA',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 8 con Normalizzazione e Information Gain
    ##############################################################
    print('Pipeline 8')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e Information Gain',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 9 con Normalizzazione e SMOTE
    ##############################################################
    print('Pipeline 9')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 10 con Normalizzazione,PCA e SMOTE
    ##############################################################
    print('Pipeline 10')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, PCA e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 11 con Normalizzazione,Information Gain e SMOTE
    ##############################################################
    print('Pipeline 11')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, Information Gain e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 12 con Standardizzazion e PCA
    ##############################################################
    print('Pipeline 12')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e PCA',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 13 con Standardizzazion e Information Gain
    ##############################################################
    print('Pipeline 13')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e Information Gain',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 14 con Standardizzazione e SMOTE
    ##############################################################
    print('Pipeline 14')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 15 con Standardizzazione,PCA e SMOTE
    ##############################################################
    print('Pipeline 15')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, PCA e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 16 con Standardizzazione,Information Gain e SMOTE
    ##############################################################
    print('Pipeline 16')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, Information Gain e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 17 con PCA e SMOTE
    ##############################################################
    print('Pipeline 17')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con PCA e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 18 con Information Gain e SMOTE
    ##############################################################
    print('Pipeline 18')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Information Gain e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 19 con Borderline-SMOTE
    ##############################################################
    print('Pipeline 19')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Boredeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 20 con Borderline-SMOTE SVM
    ##############################################################
    print('Pipeline 20')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con SVMSMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 21 con Adaptive Synthetic Sampling (ADASYN)
    ##############################################################
    print('Pipeline 21')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 22 con Normalizzazione e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 22')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 23 con Normalizzazione,PCA e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 23')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, PCA e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 24 con Normalizzazione,Information Gain e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 24')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, Information Gain e Bordeline_SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 25 con Standardizzazione e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 25')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 26 con Standardizzazione,PCA e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 26')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, PCA e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 27 con Standardizzazione,Information Gain e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 27')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, Information Gain e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 28 con PCA e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 28')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con PCA e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 29 con Information Gain e Bordeline-SMOTE
    ##############################################################
    print('Pipeline 29')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('Bordeline_SMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Information Gain e Bordeline-SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 30 con Normalizzazione e SMOTESVM
    ##############################################################
    print('Pipeline 30')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 31 con Normalizzazione,PCA e SMOTESVM
    ##############################################################
    print('Pipeline 31')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, PCA e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 32 con Normalizzazione,Information Gain e SMOTESVM
    ##############################################################
    print('Pipeline 32')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, Information Gain e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 33 con Standardizzazione e SMOTESVM
    ##############################################################
    print('Pipeline 33')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 34 con Standardizzazione,PCA e SMOTESVM
    ##############################################################
    print('Pipeline 34')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, PCA e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 35 con Standardizzazione,Information Gain e SMOTESVM
    ##############################################################
    print('Pipeline 35')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, Information Gain e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 36 con PCA e SMOTESVM
    ##############################################################
    print('Pipeline 36')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con PCA e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 37 con Information Gain e SMOTESVM
    ##############################################################
    print('Pipeline 37')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Information Gain e SMOTESVM',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 38 con Normalizzazione e ADASYN
    ##############################################################
    print('Pipeline 38')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 39 con Normalizzazione,PCA e ADASYN
    ##############################################################
    print('Pipeline 39')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('SVMSMOTE',{
        'strategy':'auto',
        'k_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, PCA e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 40 con Normalizzazione,Information Gain e ADASYN
    ##############################################################
    print('Pipeline 32')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('normalization',{'norm':'max'}))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Normalizzazione, Information Gain e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 41 con Standardizzazione e ADASYN
    ##############################################################
    print('Pipeline 41')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )


    ##############################################################
    # Pipeline 42 con Standardizzazione,PCA e SMOTESVM
    ##############################################################
    print('Pipeline 42')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, PCA e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 43 con Standardizzazione,Information Gain e ADASYN
    ##############################################################
    print('Pipeline 43')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('standardization',None))
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Standardizzazione, Information Gain e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 44 con PCA e ADASYN
    ##############################################################
    print('Pipeline 44')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.95}))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con PCA e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )

    ##############################################################
    # Pipeline 45 con Information Gain e ADASYN
    ##############################################################
    print('Pipeline 45')
    list_preProcessing_Pipeline.clear()
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.05}))
    list_preProcessing_Pipeline.append(('ADASYN',{
        'strategy':'auto',
        'n_neighbors':5,
        'random_state':42
    }))

    list_evaluated_method.clear()
    list_evaluated_method.append(('cross_validation',{'cv':10,}))

    clf=KNeighborsClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Information Gain e ADASYN',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )
















