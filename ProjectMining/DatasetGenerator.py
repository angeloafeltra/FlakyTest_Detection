import pandas as pd
from Project import Project
from colorama import Fore
from colorama import Style
def convertStringToList(str):
    list_test=[]
    for char in "[]' ":
        str=str.replace(char,"")


    str_split=str.split(',')
    for test in str_split:
        list_test.append(test)
    return list_test


def getProjectWithMultiCommit(list_project):
    list_project_copy=list_project
    #Essendo che i nomi dei progetti commit differenti sono diversi, vengono seguiti dal SHA, li rendo tutti uguali
    for project in list_project_copy:
        project.setProjectName(project.getProjectName().split("(")[0])

    list_project_with_more_commit=[]
    for project in list_project_copy:
        count=0;
        if project.getProjectName() not in list_project_with_more_commit:
            for project2 in list_project_copy:
                if project.getProjectName()==project2.getProjectName():
                    count=count+1
            if count>=2:
                list_project_with_more_commit.append(project.getProjectName())

    for project in list_project_with_more_commit:
        print(project)

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
        list_project.append(Project(splits[0],splits[2],splits[3],convertStringToList(splits[4])))




    ###############################################################
    # Creo un dataset per ogni repository e un dataset generale
    ###############################################################
    fp=open("DatasetInfo.txt",'a')
    fp.write('Repository;PathDataset;N_TestNonFlaky;N_TestFlaky;N_TestFlakyPersi\n');

    # Path Assoluto della cartella in cui sono stati salvati i file con le metriche estratte
    PATH_CARTELLA = 'D:\\Universita\\FlakyPaper\\DataSet\\gitClone\\'
    datasetGeneraleCreato=False
    numeroTotale_TestFlakyPersi=0


    for project in list_project:
        dataset=pd.read_csv(PATH_CARTELLA+project.getProjectName())
        dataset['isFlaky']=0

        for test in project.getListTestFlaky():
            dataset.loc[dataset['testCase']==test,'isFlaky']=1

        pathDataset = 'D:\\Universita\\FlakyPaper\\DataSet\\CSV\\' + project.getProjectName() + '.csv'
        dataset.to_csv(pathDataset, index=None)

        testFlaky = (dataset['isFlaky'] == 1).sum()
        testNonFlaky = (dataset['isFlaky'] == 0).sum()
        testFlakyPersi = len(project.getListTestFlaky()) - testFlaky
        numeroTotale_TestFlakyPersi=numeroTotale_TestFlakyPersi+testFlakyPersi



        ##############################################################################
        # Eseguo un controllo per vedere se i test flaky persi
        # sono presenti nel file contenente la lista dei test scartati dal tool
        ##############################################################################
        fp2=open('D:\\Universita\\FlakyPaper\\DataSet\\gitClone\\'+project.getProjectName()+'TestReject','r')
        list_project_reject = []
        i = 0
        for row in fp2:
            if not i == 0:
                splits = row.split(',')
                list_project_reject.append(splits[2].replace('\n', ""))
            else:
                i = i + 1
        test_flakyReject=0
        for test in project.getListTestFlaky():
            if test.replace('\n', "") in list_project_reject:
                test_flakyReject = test_flakyReject+1


        if test_flakyReject!=testFlakyPersi:
            print(Fore.RED + "I test flaky persi e il numero dei test flaky presenti nella lista dei test reject non coincide per il progetto {}: {}!={}".format(project.getProjectName(),testFlakyPersi,test_flakyReject) + Style.RESET_ALL)
        fp2.close()
        fp.write(project.getProjectName() + ';' +pathDataset+';'+str(testNonFlaky)+';'+str(testFlaky)+';'+str(testFlakyPersi)+'\n')
        print("Flaky Test {} Persi:{}".format(project.getProjectName(), testFlakyPersi))


        if datasetGeneraleCreato==False:
            df_globale=dataset
            datasetGeneraleCreato=True
        else:
            df_globale=pd.concat([df_globale,dataset],axis=0)


    print("Test Non Flaky:{},Test Flaky:{}".format((df_globale['isFlaky'] == 0).sum(),(df_globale['isFlaky'] == 1).sum()))
    df_globale.to_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale.csv', index=None)
    print("Numero totale test flaky persi:{}".format(numeroTotale_TestFlakyPersi))
    fp.close()




