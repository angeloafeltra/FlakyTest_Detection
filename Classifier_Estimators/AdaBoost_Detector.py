from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import  Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost_Detector:
    #Pipeline Normalizzazione e SMOTE

    def __init__(self,k_neighbors):
        self.norm=Normalizer(norm='max')
        if k_neighbors>4:
            k_neighbors=5
        self.smote=SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
        self.dt_ada=DecisionTreeClassifier(criterion='entropy',splitter='best')
        self.ada=AdaBoostClassifier(base_estimator=self.dt_ada,n_estimators=150,learning_rate=0.1,algorithm='SAMME.R')

    def fit(self,X_set,y_set):
        X_set=X_set.to_numpy()
        y_set=y_set.to_numpy()
        X_set=self.norm.fit_transform(X=X_set)
        X_set,y_set=self.smote.fit_resample(X=X_set,y=y_set)
        self.ada.fit(X=X_set,y=y_set)

    def predict(self,X_set):
        X_set=X_set.to_numpy()
        X_set=self.norm.transform(X=X_set)
        predict_ada=self.ada.predict(X=X_set)
        return predict_ada