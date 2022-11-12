from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier
from Plot import Plot
import numpy as np
import mlflow
import pandas

class Data_PreProcessing:

    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.columnsName_X=self.X.columns
        self.columnsName_Y=self.y.name
        self.generatorePlot=Plot()

    def getX(self): return self.X
    def setX(self,X):
        self.X=X
        self.columnsName_X=self.X.columns

    def getY(self): return self.y
    def setY(self,y):
        self.y=y
        self.columnsName_Y=self.y.name

    def normalization(self,norm):
        normalizer=Normalizer(norm=norm)
        self.X=normalizer.fit_transform(X=self.__getNumpyX())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        return normalizer

    def standardization(self):
        standardScaler=StandardScaler()
        self.X=standardScaler.fit_transform(X=self.__getNumpyX())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        return standardScaler


    def PCA(self,varianza_comulativa):
        cov_mat=np.cov(self.X.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plot=self.generatorePlot.print_varianza_comulativa("Varianza Feature PCA",cum_var_exp,var_exp,len(self.columnsName_X))
        mlflow.log_figure(plot,"Varianza Feature PCA.png")
        i=1
        n_components=0
        for cum_var in cum_var_exp:
            if cum_var>=varianza_comulativa:
                n_components=i
            else:
                i=i+1
        pca=PCA(n_components=n_components)
        self.X=pca.fit_transform(X=self.__getNumpyX())
        self.columnsName_X=[]
        for n in range(0,n_components):
            self.columnsName_X.append('PCA_Component_'+str(n))
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        plot=self.generatorePlot.print_DataFrame(title="Dataset dopo PCA",X=self.X,y=self.y)
        mlflow.log_param("PCA components",n_components)
        mlflow.log_metric("Varianza coumlativa",cum_var_exp[n_components-1])
        mlflow.log_figure(plot,"Dataset dopo PCA.png")
        return pca


    def information_gain(self,threshold):
        self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        rf_fs=RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)
        rf_fs.fit(X=self.X,y=self.y)
        importance=rf_fs.feature_importances_
        indices=np.argsort(importance)[::-1]
        colum_remove=[]
        for f in range (self.X.shape[1]):
            if importance[indices[f]] < threshold:
                colum_remove.append(self.columnsName_X[indices[f]])

        plot=self.generatorePlot.print_fetaureImportances(self.X,importance,indices,self.columnsName_X)
        mlflow.log_figure(plot,"Feature Importances.png")
        self.X=self.X.drop(colum_remove,axis=1)
        self.columnsName_X=self.X.columns
        self.X=self.X.to_numpy()

        mlflow.log_param("IG_Threshold",threshold)
        mlflow.log_metric("IG_FeatureRimosse",len(colum_remove))

        return colum_remove



    def SMOTE(self, strategy, k_neighbors, random_state):
        plot=self.generatorePlot.print_DataFrame(title="Dataset Non Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Non Bilanciato.png")
        smote=SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=random_state)
        self.X,self.y=smote.fit_resample(X=self.__getNumpyX(),y=self.__getNumpyY())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        #self.y=pandas.Series(self.y,name=self.columnsName_Y)
        plot=self.generatorePlot.print_DataFrame(title="Dataset Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Bilanciato.png")
        mlflow.log_param("SMOTE_Strategy",strategy)
        mlflow.log_param("SMOTE_K", k_neighbors)

    def Bordeline_SMOTE(self, strategy, k_neighbors, random_state):
        plot=self.generatorePlot.print_DataFrame(title="Dataset Non Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Non Bilanciato.png")
        smote=BorderlineSMOTE(sampling_strategy=strategy,k_neighbors=k_neighbors,random_state=random_state)
        self.X,self.y=smote.fit_resample(X=self.__getNumpyX(),y=self.__getNumpyY())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        #self.y=pandas.Series(self.y,name=self.columnsName_Y)
        plot=self.generatorePlot.print_DataFrame(title="Dataset Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Bilanciato.png")
        mlflow.log_param("SMOTE_Strategy",strategy)
        mlflow.log_param("SMOTE_K", k_neighbors)

    def SMOTENC(self, strategy, k_neighbors, random_state):
        plot=self.generatorePlot.print_DataFrame(title="Dataset Non Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Non Bilanciato.png")
        smote=SMOTENC(sampling_strategy=strategy,k_neighbors=k_neighbors,random_state=random_state)
        self.X,self.y=smote.fit_resample(X=self.__getNumpyX(),y=self.__getNumpyY())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        #self.y=pandas.Series(self.y,name=self.columnsName_Y)
        plot=self.generatorePlot.print_DataFrame(title="Dataset Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Bilanciato.png")
        mlflow.log_param("SMOTE_Strategy",strategy)
        mlflow.log_param("SMOTE_K", k_neighbors)

    def SVMSMOTE(self, strategy, k_neighbors, random_state):
        plot=self.generatorePlot.print_DataFrame(title="Dataset Non Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Non Bilanciato.png")
        smote=SVMSMOTE(sampling_strategy=strategy,k_neighbors=k_neighbors,random_state=random_state)
        self.X,self.y=smote.fit_resample(X=self.__getNumpyX(),y=self.__getNumpyY())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        #self.y=pandas.Series(self.y,name=self.columnsName_Y)
        plot=self.generatorePlot.print_DataFrame(title="Dataset Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Bilanciato.png")
        mlflow.log_param("SMOTE_Strategy",strategy)
        mlflow.log_param("SMOTE_K", k_neighbors)

    def ADASYN(self, strategy, n_neighbors, random_state):
        plot=self.generatorePlot.print_DataFrame(title="Dataset Non Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Non Bilanciato.png")
        smote=ADASYN(sampling_strategy=strategy,n_neighbors=n_neighbors,random_state=random_state)
        self.X,self.y=smote.fit_resample(X=self.__getNumpyX(),y=self.__getNumpyY())
        #self.X=pandas.DataFrame(self.X,columns=self.columnsName_X)
        #self.y=pandas.Series(self.y,name=self.columnsName_Y)
        plot=self.generatorePlot.print_DataFrame(title="Dataset Bilanciato",X=self.X,y=self.y)
        mlflow.log_figure(plot,"Dataset Bilanciato.png")
        mlflow.log_param("SMOTE_Strategy",strategy)
        mlflow.log_param("SMOTE_K", n_neighbors)

    def __getNumpyX(self):
        if not isinstance(self.X, np.ndarray):
            self.X=self.X.to_numpy()

        return self.X

    def __getNumpyY(self):
        if not isinstance(self.y,np.ndarray):
            self.y=self.y.to_numpy()

        return self.y