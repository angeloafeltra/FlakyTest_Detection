
class Project:

    def __init__(self,projectName,urlProject,SHA,listTestFlaky):
        self.projectName=projectName
        self.urlProject=urlProject
        self.SHA=SHA
        self.listTestFlaky=listTestFlaky

    def getProjectName(self): return self.projectName

    def setProjectName(self,projectName): self.projectName=projectName

    def getUrlProject(self): return self.urlProject

    def setUrlProject(self,urlProject): self.urlProject=urlProject

    def getSHA(self): return self.SHA

    def setSHA(self,SHA): self.SHA=SHA

    def getListTestFlaky(self): return self.listTestFlaky

    def setListTestFlaky(self,listTestFlaky): self.listTestFlaky=listTestFlaky

    def addTestFlaky(self,testFlaky):
        if not testFlaky in self.listTestFlaky:
            self.listTestFlaky.append(testFlaky)
