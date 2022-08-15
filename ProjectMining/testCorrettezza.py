import pandas as pd
from Project import Project
from colorama import Fore
from colorama import Style
def convertStringToList(str):
    list_test=[]
    for char in "[]' ":
        str=str.replace(char,"")
    str = str.replace('\n', "") #MODIFICATO

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
    # Creo un dataset per la repository
    ###############################################################
    PATH_CARTELLA = 'D:\\Universita\\FlakyPaper\\DataSet\\gitClone\\'


    for project in list_project:
        print("Project: {}".format(project.getProjectName()))
        dataset=pd.read_csv(PATH_CARTELLA+project.getProjectName())
        dataset['isFlaky']=0 #Aggiungo la colonna con la lable

        for test in project.getListTestFlaky():
            dataset.loc[dataset['testCase']==test,'isFlaky']=1

        pathDataset = 'D:\\Universita\\FlakyPaper\\DataSet\\CSV\\' + project.getProjectName() + '.csv'
        dataset.to_csv(pathDataset, index=None) #Salvo il dataset

        numTestFlaky = (dataset['isFlaky'] == 1).sum()
        numTestFlakyPersi = len(project.getListTestFlaky()) - numTestFlaky

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
        fp2.close()


        numTest_flakyReject=0
        for test in project.getListTestFlaky():
            if test in list_project_reject:
                numTest_flakyReject = numTest_flakyReject+1

        if numTest_flakyReject!=numTestFlakyPersi:
            print(Fore.RED + "I test flaky persi e il numero dei test flaky reject non coincide per il progetto {}: {}!={}".format(project.getProjectName(),numTestFlakyPersi,numTest_flakyReject) + Style.RESET_ALL)
            print(project.getListTestFlaky())
            print(len(project.getListTestFlaky()))
            break;







