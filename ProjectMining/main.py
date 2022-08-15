import pandas as pd
from Project import Project



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
    #Essendo che i nomi dei progetti con piu commit diversi li rendo tutti uguali
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

    return list_project_with_more_commit

if __name__ == '__main__':
    fp=open('DatasetInfo.txt','r')



    fp2=open('RepositoryClonate.txt','r')
    list_project=[]
    for row in fp2:
        splits=row.split(';')
        list_project.append(Project(splits[0],splits[2],splits[3],convertStringToList(splits[4])))

    list_project_with_multiCommit=getProjectWithMultiCommit(list_project)



    datasetGeneraleCreato = False
    i = 0
    nomeRepository=None
    datasetRepositoryConPiuCommit=None
    numero_testFlakyRepositoryConPiuCommit=0
    conteggioRepository=0
    for row in fp:
        if not i == 0: #Condizione per evitare la riga d'intestazione
            row_parts = row.split(';')
            if row_parts[0].split('(')[0] in list_project_with_multiCommit: #Verifico se la repository possiede piu commit, salvando quella con piu test flaky
                if numero_testFlakyRepositoryConPiuCommit<=int(row_parts[3]):
                    datasetRepositoryConPiuCommit=row_parts[1]
                    numero_testFlakyRepositoryConPiuCommit=int(row_parts[3])
                    nomeRepository=row_parts[0].split('(')[0]
            else:
                if not nomeRepository==row_parts[0] and not nomeRepository is None: #E un altro progetto quindi concateno al dataset generale il progetto precedentemente tracciato
                    if numero_testFlakyRepositoryConPiuCommit>0:
                        dataset = pd.read_csv(datasetRepositoryConPiuCommit)
                        if datasetGeneraleCreato == False:
                            datasetGenerale = dataset
                            datasetGeneraleCreato = True
                            conteggioRepository=conteggioRepository+1
                        else:
                            datasetGenerale = pd.concat([datasetGenerale, dataset], axis=0)
                            conteggioRepository = conteggioRepository + 1
                    nomeRepository = None
                    datasetRepositoryConPiuCommit = None
                    numero_testFlakyRepositoryConPiuCommit = 0

                if not row_parts[3] == '0':
                    dataset = pd.read_csv(row_parts[1])
                    if datasetGeneraleCreato == False:
                        datasetGenerale = dataset
                        datasetGeneraleCreato = True
                        conteggioRepository = conteggioRepository + 1
                    else:
                        datasetGenerale = pd.concat([datasetGenerale, dataset], axis=0)
                        conteggioRepository = conteggioRepository + 1
        else:
            i = i + 1

    print(datasetGenerale.info())
    print("Test Non Flaky:{},Test Flaky:{}".format((datasetGenerale['isFlaky'] == 0).sum(),
                                                   (datasetGenerale['isFlaky'] == 1).sum()))


    datasetGenerale.to_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale2.csv', index=None)
    fp.close()

    print(conteggioRepository)