
from pydriller import Repository, Git
from git import Repo
import os
import pandas
from Project import Project


DATASET_NAME = 'Flaky Tests_All Projects_Tabella.csv'


def loadingDataSet(datasetname):
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)



def extractProjectsToDataFrame(dataset):
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


def convertStringToList(str):
    list_test=[]
    for char in "[]' ":
        str=str.replace(char,"")


    str_split=str.split(',')
    for test in str_split:
        list_test.append(test)
    return list_test

if __name__ == '__main__':

    ##############################################
    # Carico il dataset esportato ed estraggo i progetti
    ##############################################
    dataset = loadingDataSet(DATASET_NAME)
    dataset = dataset[['Project URL', 'SHA Detected', 'Fully-Qualified Test Name (packageName.ClassName.methodName)']]
    list_projects=extractProjectsToDataFrame(dataset)



    ##############################################
    # Eseguo un analisi dei dati esportati
    ##############################################
    print("Numero Progetti:{}".format(len(list_projects)))
    numero_testFlaky=0
    for project in list_projects:
        numero_testFlaky=numero_testFlaky+len(project.getListTestFlaky())

    print("Numero Test Flaky:{}".format(numero_testFlaky))
    print("Test Flaky Duplicati:")
    df_testFlaky = dataset[['Fully-Qualified Test Name (packageName.ClassName.methodName)','Project URL']]
    duplicati = df_testFlaky[df_testFlaky.duplicated()]

    for test, url in zip(duplicati['Fully-Qualified Test Name (packageName.ClassName.methodName)'],duplicati['Project URL']):
        print('{} : {}'.format(test,url))



    ##############################################
    # Eseguo il clone e il checkout dei progetti
    ##############################################

    # Apro un file per tracciare i file clonati correttamente e uno per tracciare i file non clonati correttamente
    fileClonati = open('RepositoryClonate.txt', 'a')
    fileNonClonatiCorrettamente = open('RepositoryNonClonateCorrettamente.txt', 'a')


    #Path Assoluto della cartella in cui clonare le respository
    PATH_CARTELLA='D:\\Universita\\\FlakyPaper\\\gitClone\\'

    for project in list_projects:
        outputPath = PATH_CARTELLA + project.getProjectName()

        #Controllo se la directory gia esiste, quindi è stato eseguito gia il clone
        if not os.path.isdir(outputPath):
            # Eseguo il clone della repo
            try:
                Repo.clone_from(project.getUrlProject(), outputPath)
                gr = Git(outputPath)
                gr.checkout(project.getSHA())
                fileClonati.write(project.getProjectName() + ';' +outputPath+';'+ project.getUrlProject() + ';' + project.getSHA() + ';' + str(project.getListTestFlaky()) + '\n')
            except Exception as e:
                print(e)
                fileNonClonatiCorrettamente.write(project.getProjectName() + ';' +outputPath+';'+ project.getUrlProject() + ';' + project.getSHA() + ';' + str(project.getListTestFlaky()) + '\n')

        else:
            print("Progetto gia clonato")


    fileClonati.close()
    fileNonClonatiCorrettamente.close()


    fileNonClonatiCorrettamente = open('RepositoryNonClonateCorrettamente.txt', 'r')
    numero_repositoryPerse=0
    numero_testFlakyPersi=0
    for row in fileNonClonatiCorrettamente:
        splits = row.split(';')
        numero_repositoryPerse=numero_repositoryPerse+1
        numero_testFlakyPersi=numero_testFlakyPersi+len(convertStringToList(splits[4]))

    print("Repository non clonate correttamente:{}".format(numero_repositoryPerse))
    print("Test Flaky Persti:{}".format(numero_testFlakyPersi))

    fileNonClonatiCorrettamente.close()
