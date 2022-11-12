import concurrent
import itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats as st
import mlflow
import pandas as pd
import re
from os.path import exists
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def calc_mode(row):
    mode=row.mode(dropna=False)
    return mode[0]


def validate_ensamble(combinazione,cont_file_salv_val,file_salvataggio_valutazioni):

    #Verifico se la combinazione è stata gia valutata
    valutata=False
    for combinazione_valutata in cont_file_salv_val:
        if combinazione.strip() in combinazione_valutata:
            valutata=True
            break
    if valutata==False:
        #Carico tutti i dataset della combinazione
        datasets_path=re.findall('(file[A-Za-z0-9:/_ .]+)',combinazione) #Con questa regex ottengo i path dei vari dataset
        datasets=[]
        dataset_predict=pd.DataFrame() #Dataset che conterra le predizioni di ogni test set
        for dataset_path,n in zip(datasets_path,range(len(datasets_path))):
            dataset=pd.read_csv(dataset_path)
            datasets.append(dataset)
            name_col='predict'+str(n)
            dataset_predict[name_col]=dataset['isFlakyPredict']

        dataset_generale=datasets[0][['idProject','nameProject','testCase','isFlaky']]
        #Calcolo la moda
        print("Calcolo la moda")
        predict = ['predict{}'.format(i) for i in range(len(datasets_path))]

        '''
        Metodo1
        mode=dataset_predict[predict].mode(axis=1, dropna=False)
        dataset_generale['isFlakyPredict']=mode[0]
        
        Medoto2
        dataset_generale['isFlakyPredict']=dataset_predict[predict].apply(calc_mode,axis=1)
        '''
        #Metodo3
        arr=dataset_predict[predict].to_numpy()
        dataset_generale['isFlakyPredict']=st.mode(arr,axis=1).mode
        print("Termine calcolo moda")

        #Salvo le performance della combinazione in una lista
        performance=[]
        print("Calcolo le predizioni")
        performance.append(accuracy_score(y_true=dataset_generale['isFlaky'], y_pred=dataset_generale['isFlakyPredict']))
        performance.append(precision_score(y_true=dataset_generale['isFlaky'], y_pred=dataset_generale['isFlakyPredict']))
        performance.append(recall_score(y_true=dataset_generale['isFlaky'], y_pred=dataset_generale['isFlakyPredict']))
        performance.append(f1_score(y_true=dataset_generale['isFlaky'], y_pred=dataset_generale['isFlakyPredict']))
        file_salvataggio_valutazioni.write(combinazione.strip()+":"+str(performance)+"\n")
    print("Terminato")


if __name__ == "__main__":
    experiments_name=['Decision Tree','Random Forest','KNN','AdaBoost']
    if exists('Combinazion_Pipeline.txt') is False: #Non possiedo le combinazioni, quindi le genero
        ##################################################
        # Genero le combinazioni
        ##################################################
        print("Genero le combinazioni")
        runs_experiments=[] #Lista che conterra una lista di run per ogni esperimento
        for experiment_name in experiments_name:
            experiment=mlflow.get_experiment_by_name(experiment_name) #Ottengo un object experiment
            all_run=mlflow.search_runs(experiment_ids=[experiment.experiment_id]) #Ottengo tutte le runs dell'esperimento (E un file csv)
            runs_list=[]
            for artifact_uri,model_history,run_name in zip(all_run['artifact_uri'],all_run['tags.mlflow.log-model.history'],all_run['tags.mlflow.runName']):
                model_artifact=re.search('\"artifact_path\": \"([A-Za-z0-9]+)\"',model_history).group(1) #Ottengo il nome del modello usato nella run
                test_set_path=artifact_uri+'/'+model_artifact+' Prediction'+'/'+model_artifact+'.csv' #Path in cui e salvato il test_set del run
                tripla=[run_name,test_set_path,model_artifact] #Creo una tripla (nome_run,path_testSet,modello_usato)
                runs_list.append(tripla)
            runs_experiments.append(runs_list)

        #Creo le varie combinazioni e le salvo in un file
        fp=open('Combinazion_Pipeline.txt','w')
        for element in itertools.product(*runs_experiments):
            fp.write(str(element)+'\n')
        fp.close()

    ########################################################
    # Valuto ogni combinazione
    #######################################################

    print("Valuto le combinazioni")
    fp=open('Combinazion_Pipeline.txt','r')
    fp2=open('Combinazion_Pipeline_Valutate.txt','r+')
    contents=fp.readlines()
    contents2=fp2.readlines()

    for combinazione,n in zip(contents,range(len(contents))):
        print("Combinazione {}".format(n))
        parametri={'combinazione': combinazione,
                   'cont_file_salv_val': contents2,
                   'file_salvataggio_valutazioni': fp2
                   }
        validate_ensamble(**parametri)


    '''
    #with ThreadPoolExecutor() as executor:
        print("Inizio l'esecuzione concorrente")
        futures=[]
        for combinazione in contents:
            parametri={'combinazione': combinazione,
                       'cont_file_salv_val': contents2,
                       'file_salvataggio_valutazioni': fp2
                       }
            #futures.append(executor.submit(validate_ensamble,combinazione=combinazione, cont_file_salv_val=contents2, file_salvataggio_valutazioni=fp2))
            futures.append(executor.submit(validate_ensamble,**parametri))
    '''
    fp2.close()


    ########################################################
    # Ordino i risultati in ordine decrescente sul f1-score
    #######################################################

    start_time = time.time()
    fp2=open('Combinazion_Pipeline_Valutate.txt','r')
    contents=fp2.readlines()
    #BubbleSort <-- Sostiuire con quick sort
    i=0
    while i<len(contents)-1:
        print('Ciclo {}'.format(i))
        j=0
        while j<len(contents)-i-1:
            f1_1=re.findall('\[[0-9.]+, [0-9.]+, [0-9.]+, ([0-9.]+)\]',str(contents[j]))
            f1_2=re.findall('\[[0-9.]+, [0-9.]+, [0-9.]+, ([0-9.]+)\]',str(contents[j+1]))
            if float(f1_1[0])>float(f1_2[0]):
                temp=contents[j]
                contents[j]=contents[j+1]
                contents[j+1]=temp
            j=j+1
        i=i+1

    fp3=open('Combinazion_Pipeline_Valutate_Ordinate.txt','w')
    for combinazione in contents:
        fp3.write(combinazione)
    fp3.close()
    fp2.close()
    print("--- %s seconds ---" % (time.time() - start_time))






