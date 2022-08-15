import pandas as pd

if __name__ == '__main__':

    dataset=pd.read_csv('D:\\Universita\\FlakyPaper\\DataSet\\DatasetGenerale2.csv')
    dataset = dataset[dataset['testCase'].str.lower().str.contains('.setup|.teardown|.before|.after') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis=1)  # Rimuovo dal dataset gli indici
    print("Test Non Flaky:{},Test Flaky:{}".format(dataset['isFlaky'].value_counts()[0],dataset['isFlaky'].value_counts()[1]))

    print(dataset.info())
    dataset_copy = dataset.drop(['idProject', 'nameProject', 'testCase'], axis=1)
    print(dataset_copy.info())
    dataset_copy = dataset_copy.drop_duplicates()
    print("Test Non Flaky:{},Test Flaky:{}".format(dataset['isFlaky'].value_counts()[0],dataset['isFlaky'].value_counts()[1]))

