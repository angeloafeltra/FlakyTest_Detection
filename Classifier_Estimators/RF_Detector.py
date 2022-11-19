from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class RF_Detector:
    #Pipeline standardizzazione e smote

    def __init__(self,k_neighbors):
        self.std=StandardScaler()
        if k_neighbors>4:
            k_neighbors=5
        self.smote=SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
        self.rf=RandomForestClassifier(n_estimators=150,criterion='entropy',class_weight=None)

    def fit(self,X_set,y_set):
        X_set=X_set.to_numpy()
        y_set=y_set.to_numpy()
        X_set=self.std.fit_transform(X=X_set)
        X_set,y_set=self.smote.fit_resample(X=X_set,y=y_set)
        self.rf.fit(X=X_set,y=y_set)

    def predict(self,X_set):
        X_set=X_set.to_numpy()
        X_set=self.std.transform(X=X_set)
        predict_rf=self.rf.predict(X=X_set)
        return predict_rf