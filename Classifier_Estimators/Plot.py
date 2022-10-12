import matplotlib
import pandas

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class Plot:

    def __int__(self):
        pass


    def print_CartesianDiagramWithMean(self,title,xLable,yLable,xAxMinValue,xAxMaxValue,yAxMinValue,yAxMaxValue,dicValue):
        plt.clf()
        plt.plot(figsize=(10,10)) #Dimensione plot
        plt.title(title) #Titolo plot
        plt.xlabel(xLable)
        plt.ylabel(yLable)
        plt.axis([xAxMinValue, xAxMaxValue, yAxMinValue, yAxMaxValue])

        for key in dicValue:
            values=dicValue.__getitem__(key)
            color= np.random.rand(3,)
            plt.plot(range(len(values)),values,marker='o',color=color,label=key)
            mean=np.mean(values)
            plt.axhline(y=mean, linestyle='--',color=color, label=key+" mean: "+str(round(mean,2)))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        return plt.gcf()


    def print_ConfusionMatrix(self,confusionMatrix):
        plt.clf()
        plt.plot(figsize=(3.5, 3.5))
        plt.matshow(confusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confusionMatrix.shape[0]):
            for j in range(confusionMatrix.shape[1]):
                plt.text(x=j, y=i, s=confusionMatrix[i, j], va='center', ha='center')
        plt.xlabel('Predict label')
        plt.ylabel('True label')

        return plt.gcf()


    def print_DataFrame(self,title,X,y):
        plt.clf()
        plt.plot(figsize=(6.4,4.8)) #Dimensione plot
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(pandas.DataFrame(X).iloc[:, 0], pandas.DataFrame(X).iloc[:, 1], marker='o', c=y,s=25, edgecolor='k', cmap=plt.cm.coolwarm)

        return plt.gcf()

    def print_varianza_comulativa(self,title,varianza_comulativa,varianza_individuale,n_feature):
        plt.clf()
        plt.title(title)
        plt.plot(figsize=(7.4,5.8)) #Dimensione plot
        plt.bar(range(1, n_feature+1), varianza_individuale, alpha=0.5, align='center', label='Varianza Individuale')
        plt.step(range(1, n_feature+1), varianza_comulativa, where='mid', label='Varianza Comulativa')
        plt.ylabel('Variance Ratio')
        plt.xlabel('Numero Componenti')
        plt.legend(loc='best')

        return plt.gcf()


    def print_fetaureImportances(self,X,importance,indices,columns):

        plt.clf()
        plt.title('Feature Importances')
        plt.plot(figsize=(7.4,5.8)) #Dimensione plot
        plt.bar(range(X.shape[1]),importance[indices],color = 'lightblue',align = 'center')
        plt.xticks(range(X.shape[1]),columns[indices], rotation = 90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        return plt.gcf()