import mlflow
import mlflow.sklearn
import pandas
import pandas as pd
import numpy as np
import Function as f
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from Plot import Plot


def print_scores_cv(scores,name_model,n_iteration):
    accuracy_mean=np.mean(scores['accuracy'])
    precision_mean=np.mean(scores['precision'])
    recall_mean=np.mean(scores['recall'])
    f1_mean=np.mean(scores['f1'])

    plt.subplots(figsize=(6.4, 4.8))
    plt.plot(range(n_iteration), scores['accuracy'], color='r', marker='o', label="accuracy")
    plt.plot(range(n_iteration), scores['precision'],color='orange', marker='o', label="precision")
    plt.plot(range(n_iteration), scores['recall'],color='g', marker='o', label="recall")
    plt.plot(range(n_iteration), scores['f1'],color='b', marker='o', label="f1")
    plt.axhline(y=accuracy_mean, color='r', linestyle='--', label="accuracy mean: "+str(round(accuracy_mean,2)))
    plt.axhline(y=precision_mean, color='orange', linestyle='--', label="precision mean: "+str(round(precision_mean,2)))
    plt.axhline(y=recall_mean, color='g', linestyle='--', label="recall mean: "+str(round(recall_mean,2)))
    plt.axhline(y=f1_mean, color='b', linestyle='--', label="f1 mean: "+str(round(f1_mean,2)))

    plt.axis([0, n_iteration-1, 0.0, 1])
    plt.title('Cross Validation '+name_model)
    plt.legend(loc="lower left")
    plt.xlabel(str(n_iteration)+'-Fold Iteration')
    plt.ylabel('Performance')

    mlflow.log_metric('Accuracy Train Set',accuracy_mean)
    mlflow.log_metric('Precision Train Set',precision_mean)
    mlflow.log_metric('Recall Train Set',recall_mean)
    mlflow.log_metric('F1-Score Train Set',f1_mean)
    mlflow.log_figure(plt.gcf(),name_model+" Nested Cross Validation.png")

def print_confusion_matrix(y_true,y_pred,estimator_name):
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predict label')
    plt.ylabel('true label')
    mlflow.log_figure(fig,estimator_name+" Confusion Matrix.png")
    mlflow.log_metric('Accuracy Test Set',accuracy_score(y_true=y_true, y_pred=y_pred))
    mlflow.log_metric('Precision Test Set',precision_score(y_true=y_true, y_pred=y_pred))
    mlflow.log_metric('Recall Test Set',recall_score(y_true=y_true, y_pred=y_pred))
    mlflow.log_metric('F1-Score Test Set',f1_score(y_true=y_true, y_pred=y_pred))

def nested_cross_validation(estimator,gridSearch_param,cv1,cv2,X,y):
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    skf=StratifiedKFold(n_splits=cv1,shuffle=False)
    grid_clf=GridSearchCV(estimator,param_grid=gridSearch_param,cv=cv2,verbose=True,n_jobs=-1,scoring='f1_micro')
    for train_index,test_index in skf.split(X=X,y=y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        grid_clf.fit(X=X_train,y=y_train)
        estimator.set_params(**grid_clf.best_params_)
        estimator.fit(X=X_train,y=y_train)
        y_predict=estimator.predict(X=X_test)
        scores['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_predict))
        scores['precision'].append(precision_score(y_true=y_test, y_pred=y_predict))
        scores['recall'].append(recall_score(y_true=y_test, y_pred=y_predict))
        scores['f1'].append(f1_score(y_true=y_test, y_pred=y_predict))

    title='Nested Cross Validation '+ estimator.__class__.__name__
    xLable=str(cv1)+'-Fold Iteration'
    yLable='Performance'
    xAxMinValue=0
    xAxMaxValue=cv1-1
    yAxMinValue=0.0
    yAxMaxValue=1
    generatePlot=Plot()
    plot=generatePlot.print_CartesianDiagramWithMean(title=title,xLable=xLable,yLable=yLable,xAxMinValue=xAxMinValue,xAxMaxValue=xAxMaxValue,yAxMinValue=yAxMinValue,yAxMaxValue=yAxMaxValue,dicValue=scores)
    #print_scores_cv(scores=scores,name_model=estimator.__class__.__name__,n_iteration=cv1)
    mlflow.log_figure(plot,title+".png")
    mlflow.log_metric('Accuracy Train Set',np.mean(scores['accuracy']))
    mlflow.log_metric('Precision Train Set',np.mean(scores['precision']))
    mlflow.log_metric('Recall Train Set',np.mean(scores['recall']))
    mlflow.log_metric('F1-Score Train Set',np.mean(scores['f1']))

def print_plot_dataset(plot_title,X,y):
    ds = pandas.DataFrame(X)
    plt.clf()
    plt.title(plot_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplots(figsize=(6.4, 4.8))
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y,s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    mlflow.log_figure(plt.gcf(),plot_title+".png")


def cross_validation(estimator,cv,X,y):
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    skf=StratifiedKFold(n_splits=cv,shuffle=False)
    for train_index,test_index in skf.split(X=X,y=y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        estimator.fit(X=X_train,y=y_train)
        y_predict=estimator.predict(X=X_test)
        scores['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_predict))
        scores['precision'].append(precision_score(y_true=y_test, y_pred=y_predict))
        scores['recall'].append(recall_score(y_true=y_test, y_pred=y_predict))
        scores['f1'].append(f1_score(y_true=y_test, y_pred=y_predict))
    print_scores_cv(scores=scores,name_model=estimator.__class__.__name__,n_iteration=cv)