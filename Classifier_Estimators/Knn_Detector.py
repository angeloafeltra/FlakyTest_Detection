from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

class Knn_Detector:
    #Pipeline solo smote

    def __init__(self,k_neighbors):
        if k_neighbors>4:
            k_neighbors=5
        self.smote=SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
        self.knn=KNeighborsClassifier(n_neighbors=30,weights='distance',algorithm='auto',metric='euclidean')

    def fit(self,X_set,y_set):
        X_set=X_set.to_numpy()
        y_set=y_set.to_numpy()
        X_set,y_set=self.smote.fit_resample(X=X_set,y=y_set)
        self.knn.fit(X=X_set,y=y_set)

    def predict(self,X_set):
        X_set=X_set.to_numpy()
        predict_knn=self.knn.predict(X=X_set)
        return predict_knn



