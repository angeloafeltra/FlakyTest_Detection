from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

class DT_Detector:
    #Pipeline PCA

    def __init__(self):
        self.pca=PCA(n_components=3)
        self.dt=DecisionTreeClassifier(criterion='entropy',splitter='best')

    def fit(self,X_set,y_set):
        X_set=X_set.to_numpy()
        y_set=y_set.to_numpy()
        X_set=self.pca.fit_transform(X=X_set)
        self.dt.fit(X=X_set,y=y_set)

    def predict(self,X_set):
        X_set=X_set.to_numpy()
        X_set=self.pca.transform(X=X_set)
        predict_dt=self.dt.predict(X=X_set)
        return predict_dt

