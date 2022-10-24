from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlflow
from Plot import Plot


class Evaluated:

    def __int__(self):
        pass

    def __init__(self,classifier,X_train,y_train):
        self.classifier=classifier
        self.X_train=X_train
        self.y_train=y_train
        self.generatePlot=Plot()


    def nested_cross_validation(self,gridSearch_param,cv1,cv2):
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        skf=StratifiedKFold(n_splits=cv1,shuffle=False)
        if not isinstance(self.X_train, np.ndarray):
            X=self.X_train.to_numpy()
        else:
            X=self.X_train
        if not isinstance(self.y_train, np.ndarray):
            y=self.y_train.to_numpy()
        else:
            y=self.y_train
        grid_clf=GridSearchCV(self.classifier,param_grid=gridSearch_param,cv=cv2,verbose=True,n_jobs=-1,scoring='f1_micro')
        for train_index,test_index in skf.split(X=X,y=y):
            X_train, X_test= X[train_index], X[test_index]
            y_train, y_test= y[train_index], y[test_index]
            grid_clf.fit(X=X_train,y=y_train)
            self.classifier.set_params(**grid_clf.best_params_)
            self.classifier.fit(X=X_train,y=y_train)
            y_predict=self.classifier.predict(X=X_test)
            scores['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_predict))
            scores['precision'].append(precision_score(y_true=y_test, y_pred=y_predict))
            scores['recall'].append(recall_score(y_true=y_test, y_pred=y_predict))
            scores['f1'].append(f1_score(y_true=y_test, y_pred=y_predict))

        title='Nested Cross Validation '+ self.classifier.__class__.__name__
        xLable=str(cv1)+'-Fold Iteration'
        yLable='Performance'
        xAxMinValue=0
        xAxMaxValue=cv1-1
        yAxMinValue=0.0
        yAxMaxValue=1
        plot=self.generatePlot.print_CartesianDiagramWithMean(title=title,xLable=xLable,yLable=yLable,xAxMinValue=xAxMinValue,xAxMaxValue=xAxMaxValue,yAxMinValue=yAxMinValue,yAxMaxValue=yAxMaxValue,dicValue=scores)
        mlflow.log_figure(plot,title+".png")
        mlflow.log_metric('Accuracy Train Set',np.mean(scores['accuracy']))
        mlflow.log_metric('Precision Train Set',np.mean(scores['precision']))
        mlflow.log_metric('Recall Train Set',np.mean(scores['recall']))
        mlflow.log_metric('F1-Score Train Set',np.mean(scores['f1']))

        return scores


    def cross_validation(self,cv):
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        skf=StratifiedKFold(n_splits=cv,shuffle=False)
        if not isinstance(self.X_train, np.ndarray):
            X=self.X_train.to_numpy()
        else:
            X=self.X_train
        if not isinstance(self.y_train, np.ndarray):
            y=self.y_train.to_numpy()
        else:
            y=self.y_train
        for train_index,test_index in skf.split(X=X,y=y):
            X_train, X_test= X[train_index], X[test_index]
            y_train, y_test= y[train_index], y[test_index]
            self.classifier.fit(X=X_train,y=y_train)
            y_predict=self.classifier.predict(X=X_test)
            scores['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_predict))
            scores['precision'].append(precision_score(y_true=y_test, y_pred=y_predict))
            scores['recall'].append(recall_score(y_true=y_test, y_pred=y_predict))
            scores['f1'].append(f1_score(y_true=y_test, y_pred=y_predict))

        title='Cross Validation '+ self.classifier.__class__.__name__
        xLable=str(cv)+'-Fold Iteration'
        yLable='Performance'
        xAxMinValue=0
        xAxMaxValue=cv-1
        yAxMinValue=0.0
        yAxMaxValue=1
        plot=self.generatePlot.print_CartesianDiagramWithMean(title=title,xLable=xLable,yLable=yLable,xAxMinValue=xAxMinValue,xAxMaxValue=xAxMaxValue,yAxMinValue=yAxMinValue,yAxMaxValue=yAxMaxValue,dicValue=scores)
        mlflow.log_figure(plot,title+".png")
        mlflow.log_metric('Accuracy Train Set',np.mean(scores['accuracy']))
        mlflow.log_metric('Precision Train Set',np.mean(scores['precision']))
        mlflow.log_metric('Recall Train Set',np.mean(scores['recall']))
        mlflow.log_metric('F1-Score Train Set',np.mean(scores['f1']))
        return scores

    def gridSearch(self,gridSearch_param,cv):
        grid_clf=GridSearchCV(self.classifier,param_grid=gridSearch_param,cv=cv,verbose=True,n_jobs=-1,scoring='f1_micro')
        grid_clf.fit(X=self.X_train,y=self.y_train)
        return grid_clf.best_params_



