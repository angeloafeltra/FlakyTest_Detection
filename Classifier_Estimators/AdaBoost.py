import copy
import mlflow
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Pipeline_Experiment import Pipeline_Experiment
import pandas as pd

DATASET_PATH = 'D:\\Universita\\FlakyTest_Detection\\DataSet\\DatasetGenerale2.csv'


if __name__ == "__main__":

    nome_classificatore='AdaBoost'
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

    # Creo un nuovo esperimento su mlflow
    #experiment_id=mlflow.create_experiment(nome_classificatore)
    experiment_id=6

    '''
    ##############################################################
    # Pipeline 1 senza data pre processing
    ##############################################################
    print('Pipeline 1')
    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    parametri_classificatore=[
        {
            'base_estimator': [dt],
            'n_estimators': [20, 50, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0],
            'algorithm':['SAMME', 'SAMME.R']
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


    clf=AdaBoostClassifier()

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

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')
    #best_params=pipeline.getBestParams()
    #clf.set_params(**best_params)


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


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
    #best_params=pipeline.getBestParams()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')

    
    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')


    clf=AdaBoostClassifier()
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
    '''
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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')

    '''
    clf=AdaBoostClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)
    '''

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
    #list_evaluated_method.append(('cross_validation',{'cv':10,}))

    dt=DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf=AdaBoostClassifier(base_estimator=dt,learning_rate=0.1,n_estimators=150,algorithm='SAMME.R')

    '''
    clf=AdaBoostClassifier()
    #best_params=pipeline.getBestParams()
    clf.set_params(**best_params)
    '''

    pipeline=Pipeline_Experiment(classifier=clf, list_preProcessing_Pipeline=list_preProcessing_Pipeline, list_evaluated_method=list_evaluated_method)
    pipeline.run_experiment(mlflow_experiment=experiment_id,
                            mlflow_run_name='Pipeline con Information Gain e SMOTE',
                            X_train_set=copy.copy(X_train_set),
                            X_test_set=copy.copy(X_test_set),
                            y_test_set=copy.copy(y_test_set),
                            y_train_set=copy.copy(y_train_set)
                            )



















