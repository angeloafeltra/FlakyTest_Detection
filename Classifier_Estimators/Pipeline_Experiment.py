import os

import mlflow
import pandas
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Evaluated import Evaluated
from Plot import Plot
from Data_PreProcessing import Data_PreProcessing
import numpy as np

class Pipeline_Experiment:

    def __init__(self, classifier, list_preProcessing_Pipeline, list_evaluated_method):
        '''
        self.X_train_set=X_train_set
        self.y_train_set=y_train_set
        self.X_test_set=X_test_set
        self.y_test_set=y_test_set
        '''
        self.classifier=classifier
        self.preProcessing_Pipeline=list_preProcessing_Pipeline
        self.evaluated_method=list_evaluated_method
        self.generatorePlot=Plot()

    def getClassifier(self): return self.classifier
    #def getClassifier_param(self): return  self.classifier_param
    def getPreProcessing_Pipeline(self): return self.preProcessing_Pipeline
    def getEvaluated_method(self): return self.evaluated_method
    def getBestParams(self): return self.best_params
    '''
    def getX_Train_set(self): return self.X_train_set
    def getY_Train_set(self): return self.y_train_set
    def getX_Test_set(self): return self.X_test_set
    def getY_Test_set(self): return self.y_test_set
    '''
    def setClassifier(self,classifier): self.classifier=classifier
    #def setClassifier_param(self,classifier_param): self.classifier_param=classifier_param
    def setPreProcessing_Pipeline(self, list_preProcessing_Pipeline): self.preProcessing_Pipeline=list_preProcessing_Pipeline
    def setEvaluated_method(self, list_evaluated_Method): self.evaluated_method=list_evaluated_Method
    '''
    def setX_Train_set(self,X_Train_set): self.X_train_set=X_Train_set
    def setY_Train_set(self,y_train_set): self.y_train_set=y_train_set
    def setX_Test_set(self,X_Test_set): self.X_test_set=X_Test_set
    def setY_Test_set(self,y_Test_set): self.y_test_set=y_Test_set
    '''

    def run_experiment(self,mlflow_experiment,mlflow_run_name,X_train_set,y_train_set,X_test_set,y_test_set):

        with mlflow.start_run(experiment_id=mlflow_experiment,run_name=mlflow_run_name):

            ############################################################
            # Eseguo le operazioni di data pre processing
            ############################################################
            list_transformer=[]
            data_preProcessing=Data_PreProcessing(X=X_train_set,y=y_train_set)
            if not self.preProcessing_Pipeline is None:
                for tupla in self.preProcessing_Pipeline:
                    func=getattr(data_preProcessing,tupla[0])
                    if not tupla[1] is None:
                        transform=func(**tupla[1])
                    else:
                        transform=func()
                    if not transform is None:
                        list_transformer.append(transform)

            X_train_set=data_preProcessing.getX()
            y_train_set=data_preProcessing.getY()

            ##############################################################
            # Valuto le prestazioni del modello
            ##############################################################
            evaluated=Evaluated(self.classifier,X_train_set,y_train_set)
            self.best_params=None
            for tupla in self.evaluated_method:
                if tupla[0] != "gridSearch":
                    func=getattr(evaluated,tupla[0])
                    func(**tupla[1])
                else:
                    func=getattr(evaluated,tupla[0])
                    self.best_params=func(**tupla[1])

            ##############################################################
            # Addestro il modello
            ##############################################################
            if not self.best_params is None:
                self.classifier.set_params(**self.best_params)
            self.classifier.fit(X=X_train_set,y=y_train_set)

            ##############################################################
            # Valido il modello
            ##############################################################
            columns=X_test_set.columns
            X_test_set=X_test_set.to_numpy()
            for transformer in list_transformer:
                if isinstance(transformer,list):
                    X_test_set=pandas.DataFrame(X_test_set,columns=columns)
                    X_test_set=X_test_set.drop(transformer,axis=1)
                    columns=X_test_set.columns
                    X_test_set=X_test_set.to_numpy()
                else:
                    X_test_set=transformer.transform(X=X_test_set)

            y_pred=self.classifier.predict(X=X_test_set)
            confmat = confusion_matrix(y_true=y_test_set['isFlaky'], y_pred=y_pred)
            #confmat = confusion_matrix(y_true=y_test_set, y_pred=y_pred)
            plot=self.generatorePlot.print_ConfusionMatrix(confusionMatrix=confmat)


            accuracy=accuracy_score(y_true=y_test_set['isFlaky'], y_pred=y_pred)
            precision=precision_score(y_true=y_test_set['isFlaky'], y_pred=y_pred)
            recall=recall_score(y_true=y_test_set['isFlaky'], y_pred=y_pred)
            f1=f1_score(y_true=y_test_set['isFlaky'], y_pred=y_pred)

            '''
            accuracy=accuracy_score(y_true=y_test_set, y_pred=y_pred)
            precision=precision_score(y_true=y_test_set, y_pred=y_pred)
            recall=recall_score(y_true=y_test_set, y_pred=y_pred)
            f1=f1_score(y_true=y_test_set, y_pred=y_pred)
            '''

            mlflow.log_figure(plot,self.classifier.__class__.__name__+" Confusion Matrix.png")
            mlflow.log_metric('Accuracy Test Set',accuracy)
            mlflow.log_metric('Precision Test Set',precision)
            mlflow.log_metric('Recall Test Set',recall)
            mlflow.log_metric('F1-Score Test Set',f1)

            mlflow.sklearn.log_model(
                sk_model=self.classifier,
                artifact_path=self.classifier.__class__.__name__
            )
            mlflow.log_params(self.classifier.get_params())
            y_test_set['isFlakyPredict']=y_pred
            y_test_set.to_csv(self.classifier.__class__.__name__+'.csv')
            mlflow.log_artifact(self.classifier.__class__.__name__+'.csv',self.classifier.__class__.__name__+' Prediction')
            os.remove(self.classifier.__class__.__name__+'.csv')
            mlflow.end_run()

            return accuracy,precision,recall,f1

