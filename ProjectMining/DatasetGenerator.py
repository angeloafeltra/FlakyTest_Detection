import pandas as pd
from Project import Project
from colorama import Fore
from colorama import Style
import Function
from sklearn.preprocessing import Normalizer

if __name__ == '__main__':

    list_project=[]
    file=open("RepositoryClonate.txt",'r')
    #Leggo il file in cui ho mantenuto traccia delle repository clonate correttamente.
    #Ogni riga del file contiene le informazioni della repository clonata e la cartella in cui e clonata
    #Utilizzando tali informazioni creo due liste:
    #   1. Lista in cui inserisco oggetti di tipo Project
    #   2. Lista in cui inserisco il path della cartella
    for row in file:
        splits=row.split(';')
        list_project.append(Project(splits[0],splits[2],splits[3],Function.convertStringToList(splits[4])))




    ###############################################################
    # Creo un dataset per ogni repository e un dataset generale
    ###############################################################
    fp=open("DatasetInfo.txt",'a')
    fp.write('Repository;PathDataset;N_TestNonFlaky;N_TestFlaky;N_TestFlakyPersi;N_TF_NonRilevatiDalDetector;Lista_TF_NonRilevatiDalDetector\n');

    # Path Assoluto della cartella in cui sono stati salvati i file con le metriche estratte
    PATH_CARTELLA = 'D:\\Universita\\FlakyPaper\\DataSet\\gitClone\\'
    datasetGeneraleCreato=False

    numeroTotale_TestFlakyPersi=0

    for project in list_project:
        dataset=pd.read_csv(PATH_CARTELLA+project.getProjectName())
        dataset['isFlaky']=0

        list_testFlaky_nonPresenti=[]
        for test in project.getListTestFlaky():
            if not dataset['testCase'].str.contains(test).any():
                list_testFlaky_nonPresenti.append(test)
            dataset.loc[dataset['testCase'].str.contains(test, case=False), 'isFlaky'] = 1

        numTestFlaky = (dataset['isFlaky'] == 1).sum()
        numTestNonFlaky = (dataset['isFlaky'] == 0).sum()
        numTestFlakyPersi = len(list_testFlaky_nonPresenti)
        numeroTotale_TestFlakyPersi = numeroTotale_TestFlakyPersi + numTestFlakyPersi

        ##############################################################################
        # Eseguo una verifica incrociate per vedere se i test flaky persi
        # sono presenti nel file contenente la lista dei test scartati dal tool
        ##############################################################################
        path_testReject='D:\\Universita\\FlakyPaper\\DataSet\\gitClone\\'+project.getProjectName()+'TestReject'
        list_project_testReject=Function.getListaTestReject(path_testReject)

        list_testNonRilevati=[]
        for test in list_testFlaky_nonPresenti:
            test_presente=False
            for testReject in list_project_testReject:
                if test in testReject:
                    test_presente = True

            if test_presente==False:
                list_testNonRilevati.append(test)

        pathDataset = 'D:\\Universita\\FlakyPaper\\DataSet\\CSV\\' + project.getProjectName() + '.csv'
        dataset.to_csv(pathDataset, index=None)
        fp.write(project.getProjectName()
                + ';'
                +pathDataset
                +';'
                +str(numTestNonFlaky)
                +';'
                +str(numTestFlaky)
                +';'
                +str(numTestFlakyPersi)
                +';'
                +str(list_testNonRilevati)
                +';'
                +str(list_testNonRilevati)
                +'\n')



        if datasetGeneraleCreato==False:
            df_globale=dataset
            datasetGeneraleCreato=True
        else:
            df_globale=pd.concat([df_globale,dataset],axis=0)


    print("Test Non Flaky:{},Test Flaky:{}".format((df_globale['isFlaky'] == 0).sum(),(df_globale['isFlaky'] == 1).sum()))

    ######################################################
    # Normalizzo il dataset generale
    ######################################################
    colums_norm=df_globale.columns[3:29]
    norm = Normalizer(norm='max')
    df_globale[colums_norm] = norm.fit_transform(X=df_globale[colums_norm])
    df_globale.to_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale.csv', index=None)
    print("Numero totale test flaky persi:{}".format(numeroTotale_TestFlakyPersi))
    fp.close()

    ######################################################################
    # Creo un ulteriore dataset generale con le seguenti caratteristiche:
    # 1. Ogni repository che compone il dataset possiede almeno un test flaky
    # 2. Per le repository con test flaky su piu commit è mantenuto solamente
    # il commit con piu test flaky
    # 3. Il dataset non contiene esempi di test di setup e teardown
    ######################################################################

    # Genero una lista dei progetti con piu di un commit
    list_project_with_moreCommit = Function.getProjectWithMultiCommit()
    # Genero una lista dei progetti con un singolo commit:
    list_project_with_singleCommit = Function.getProjectWithSingleCommit()

    for project in list_project_with_moreCommit:
        # Identifico il commit con piu test flaky
        project_with_moreTF = Function.getCommitWithMoreTF(project)
        list_project_with_singleCommit.append(project_with_moreTF)

    # Creo il dataset generale
    datasetGeneraleCreato = False
    datasetGenerale = None
    dataset = None
    conteggioRepository = 0
    for project in list_project_with_singleCommit:
        project_parts = project.split(';')
        if int(project_parts[3]) > 0:
            conteggioRepository = conteggioRepository + 1
            if datasetGeneraleCreato == True:
                dataset = pd.read_csv(project_parts[1])
                datasetGenerale = pd.concat([datasetGenerale, dataset])
            else:
                datasetGenerale = pd.read_csv(project_parts[1])
                datasetGeneraleCreato = True


    #Rimuovo i test di setup,teardown e eventuali duplicati
    datasetGenerale = datasetGenerale[datasetGenerale['testCase'].str.lower().str.contains('.setup|.teardown|.before|.after') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    datasetGenerale = datasetGenerale.reset_index()
    datasetGenerale = datasetGenerale.drop(['index'], axis=1)  # Rimuovo dal dataset gli indici

    ddatasetGenerale = datasetGenerale.drop_duplicates()

    print("Test Non Flaky:{},Test Flaky:{}".format((datasetGenerale['isFlaky'] == 0).sum(),
                                                   (datasetGenerale['isFlaky'] == 1).sum()))


    ########################################
    # Normalizzo il dataset
    ########################################
    colums_norm =  datasetGenerale.columns[3:29]
    datasetGenerale[colums_norm] = norm.fit_transform(X= datasetGenerale[colums_norm])
    datasetGenerale.to_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale2.csv', index=None)
    print(conteggioRepository)




