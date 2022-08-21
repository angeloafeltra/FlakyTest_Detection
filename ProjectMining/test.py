import Function
import pandas as pd

if __name__ == '__main__':
    dataset=pd.read_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale.csv')
    Function.printBloxPlot(dataset)