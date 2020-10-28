from sklearn import svm 
import pandas as pd

DATASET_PATH = "../../../datasets/titanic/"

class Dataset(object):
    def __init__(self):
        self.training_set = pd.read_csv(DATASET_PATH + "train.csv", sep=",", header=None)
        self.__select_features()
        
    def __select_features(self):
        print("\tSelecting features...")
        print(dir(self.training_set))

class SVMModel(Dataset):
    def __init__(self):
        super().__init__()
        
    def train(self):
        pass

    def test():
        print(svm.__doc__)