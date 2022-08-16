from Project import Project
import pandas


def convertStringToList(str):
    list_test=[]
    for char in "[]' ":
        str=str.replace(char,"")
    str=str.replace('\n',"")

    str_split=str.split(',')
    for test in str_split:
        list_test.append(test)
    return list_test




def getProjectWithMultiCommit():
    '''
    getProjectWithMultiCommit:
    1. Leggo il file dataset info
    2. Se nel nome de progetto è presente il carattere ( allora la repository ha piu commit con test flaky

    :return:
    '''

    fp = open('DatasetInfo.txt', 'r')
    fp_text = fp.readlines()
    fp.close()

    list_project_with_moreCommit = []
    for row in fp_text[1:]:
        row_parts = row.split(';')
        if '(' in row_parts[0]:
            if not row_parts[0].split('(')[0] in list_project_with_moreCommit:
                list_project_with_moreCommit.append(row_parts[0].split('(')[0])

    return list_project_with_moreCommit




def getProjectWithSingleCommit():
    '''
    getProjectWithSingleCommit:
    1. Ottengo i progetti con piu commit
    2. Leggo il file dataset info
    3. Per ogni progetto all'interno del file, se la repository non è presente
       nella lista dei progetti con piu commit, l'aggiungo alla lista dei
       progetti con un singolo commit

    :return:
    '''

    list_project_with_moreCommit=getProjectWithMultiCommit()

    fp = open('DatasetInfo.txt', 'r')
    fp_text = fp.readlines()
    fp.close()

    list_project_with_singleCommit = []
    for row in fp_text[1:]:
        row_parts = row.split(';')
        if not '(' in row_parts[0] and not row_parts[0] in list_project_with_moreCommit:
            list_project_with_singleCommit.append(row)

    return list_project_with_singleCommit



def getCommitWithMoreTF(project):
    '''
    getCommitWithMoreTF:
    1. Leggo il file datasetinfo.txt
    2. Ottengo tutti i vari commit del progetto
    3. Identifico tra i vari commit quello con piu test flaky

    :param project:
    :return:
    '''
    fp = open('DatasetInfo.txt', 'r')
    fp_text = fp.readlines()
    fp.close()

    list_more_commit = []
    for row in fp_text[1:]:
        row_parts = row.split(';')
        if project in row_parts[0]:
            list_more_commit.append(row)

    num_testFlaky = list_more_commit[0].split(';')[3]
    project_with_moreTF = list_more_commit[0]
    for project in list_more_commit[1:]:
        project_parts = project.split(';')
        if num_testFlaky < project_parts[3]:
            num_testFlaky = project_parts[3]
            project_with_moreTF = project

    return project_with_moreTF




def extractProjectsToDataFrame(dataset):
    '''
    extractProjectsToDataFrame:
    1. Per ogni riga del dataset:
    2.  Ricavo il nome del progetto dal url della repository
    3.  Se la lista dei progetti è vuota:
    4.      Inserisco nella lista un oggetto di tipo Project
    5.  Altrimenti:
    6.      Verifico se il progetto è gia presente nella lista,
            in caso contraro aggiungo un nuovo oggetto Project
            alla lista

    :param dataset:
    :return:
    '''

    list_projects=[]
    for projectUrl,SHA,test in zip(dataset['Project URL'], dataset['SHA Detected'],dataset['Fully-Qualified Test Name (packageName.ClassName.methodName)']):
        projectName=projectUrl.split('/')[-1]
        if len(list_projects)==0: #La lista dei progetti e vuota, quindi inserisco un progetto
            list_projects.append(Project(projectName,projectUrl,SHA,[test]))
        else:
            project=checkAddNewProject(projectName,projectUrl,SHA,test,list_projects) #La lista dei progetti non è vuota, quindi controllo se iniserire il progetto
            if not project is None:
                list_projects.append(project)

    return list_projects




def checkAddNewProject(projectName,projectURL,SHA,test,list_projects):
    '''
    checkAddNerProject:
    1. Per ogni progetto nella lista
    2.  Se l'URL del progetto è uguale al parametro projectURL:
    3.      Il progetto è gia presente nella lista
    4.      Se l'SHA del progetto è uguale al parametro SHA:
    5.          Devo aggiungere solamente il test alla lista dei test flaky del progetto
    6.          Ritorno None per indicare di non aggiungere un nuovo progetto

    7.  Se il progetto è gia presente ma ha un SHA diverso:
    8.       Ritorno un nuovo oggetto Project aggiungendo al project name l'SHA
    9.   Altrimenti (L'unico caso rimanente è quello che non è presente il progetto nella lista)
    10.      Ritorno un nuovo oggetto Project

    :param projectName:
    :param projectURL:
    :param SHA:
    :param test:
    :param list_projects:
    :return:
    '''
    progettoPresente=False
    for project in list_projects:
        if project.getUrlProject() == projectURL: #Progetto presente
            progettoPresente=True
            if project.getSHA() == SHA: #Commit presente
                project.addTestFlaky(test)
                return None

    if progettoPresente: #Progetto presente ma con commit differente
        return Project(projectName+'('+SHA+')',projectURL,SHA,[test])
    else: #Progetto non presente
        return Project(projectName,projectURL,SHA,[test])




def getListaTestReject(pathFile):
    fp2 = open(pathFile, 'r',encoding="utf8")
    list_project_testReject = []
    i = 0
    for row in fp2:
        if not i == 0:
            splits = row.split(',')
            list_project_testReject.append(splits[2].replace('\n', ""))
        else:
            i = i + 1
    fp2.close()
    return list_project_testReject

