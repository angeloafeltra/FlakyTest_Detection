import copy

import pandas as pd
from scipy import stats as st
from DT_Detector import DT_Detector
from RF_Detector import RF_Detector
from Knn_Detector import Knn_Detector
from AdaBoost_Detector import AdaBoost_Detector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
from Plot import Plot


class FlakyTestDetector:
    """
    Ensemble:
        - Decision Tree (PCA)
        - Random Forest (Standardizzazione e SMOTE)
        - Knn (SMOTE)
        - AdaBoost (Normalizzazione e SMOTE)
    """

    def __init__(self,k_neighbors):
        self.dt=DT_Detector()
        self.rf=RF_Detector(k_neighbors)
        self.knn=Knn_Detector(k_neighbors)
        self.ada=AdaBoost_Detector(k_neighbors)
        self.generatorePlot=Plot()


    def fit(self,dataset):
        X_train_set=dataset.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_train_set=dataset['isFlaky']
        self.dt.fit(X_set=copy.copy(X_train_set),y_set=copy.copy(y_train_set))
        self.rf.fit(X_set=copy.copy(X_train_set),y_set=copy.copy(y_train_set))
        self.knn.fit(X_set=copy.copy(X_train_set),y_set=copy.copy(y_train_set))
        self.ada.fit(X_set=copy.copy(X_train_set),y_set=copy.copy(y_train_set))


    def predict(self,dataset):
        X_test_set=dataset.drop(['idProject','nameProject','testCase','isFlaky'],axis=1)
        y_test_set=dataset[['idProject','nameProject','testCase','isFlaky']]
        dataset_predict=pd.DataFrame() #Dataset che conterra le predizioni di ogni test set
        dataset_predict['predict0']=self.dt.predict(X_set=copy.copy(X_test_set))
        dataset_predict['predict1']=self.rf.predict(X_set=copy.copy(X_test_set))
        dataset_predict['predict2']=self.knn.predict(X_set=copy.copy(X_test_set))
        dataset_predict['predict3']=self.ada.predict(X_set=copy.copy(X_test_set))
        #Ensamble predict
        predict = ['predict{}'.format(i) for i in range(4)]
        arr=dataset_predict[predict].to_numpy()
        predict_ensamble=st.mode(arr,axis=1).mode
        accuracy=accuracy_score(y_true=y_test_set['isFlaky'], y_pred=predict_ensamble)
        precision=precision_score(y_true=y_test_set['isFlaky'], y_pred=predict_ensamble)
        recall=recall_score(y_true=y_test_set['isFlaky'], y_pred=predict_ensamble)
        f1=f1_score(y_true=y_test_set['isFlaky'], y_pred=predict_ensamble)

        confmat = confusion_matrix(y_true=y_test_set['isFlaky'], y_pred=predict_ensamble)
        #confmat = confusion_matrix(y_true=y_test_set, y_pred=y_pred)
        plot=self.generatorePlot.print_ConfusionMatrix(confusionMatrix=confmat)
        return accuracy,precision,recall,f1,confmat,plot






