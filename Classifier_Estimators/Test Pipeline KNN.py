import copy
import mlflow
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from Pipeline_Experiment import Pipeline_Experiment
import pandas as pd

DATASET_PATH = 'D:\\Universita\\Flaky-Tests-Learning-Pipeline\\datasetFlakyTest.csv'

def get_object_colum(dataset):
    drop_col = []
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            drop_col.append(col)
    return drop_col

if __name__ == "__main__":

    nome_classificatore='Test Pipeline KNN 2'
    best_params=None
    list_evaluated_method=[]
    list_preProcessing_Pipeline=[]


    dataset_flaky = pd.read_csv(DATASET_PATH)
    dataset_copy=dataset_flaky.copy()
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False] #Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy=dataset_copy.drop_duplicates()
    dataset_copy=dataset_copy.reset_index()
    dataset_copy=dataset_copy.drop(['Unnamed: 0','index'],axis=1) #Rimuovo dal dataset gli indici

    # Divido il dataset in train set e test set
    stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratifiedShuffleSplit.split(X=dataset_copy, y=dataset_copy['isFlaky']):
        train_set = dataset_copy.loc[train_index]
        test_set = dataset_copy.loc[test_index]






    train_set_copy=train_set.copy() #Lavoro sempre su una copia del train set
    #1. Rimuovo dal dataset le feature che sono object
    train_set_copy=train_set_copy.drop(get_object_colum(train_set_copy),axis=1)
    #2. Divido le etichette dai campioni converto tutto in un array numpy
    X_train_set=train_set_copy.drop(['isFlaky'],axis=1)
    y_train_set= train_set_copy['isFlaky']

    test_set_copy=test_set.copy()
    test_set_copy=test_set_copy.drop(get_object_colum(test_set_copy),axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1)
    y_test_set = test_set_copy['isFlaky']

    # Creo un nuovo esperimento su mlflow
    experiment_id=mlflow.create_experiment(nome_classificatore)


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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))

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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))

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
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))

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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))

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
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))

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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))

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
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('PCA',{'varianza_comulativa':0.98}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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
    list_preProcessing_Pipeline.append(('information_gain',{'threshold':0.02}))
    list_preProcessing_Pipeline.append(('SMOTE',{
        'strategy':'auto',
        'k_neighbors':3,
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



















