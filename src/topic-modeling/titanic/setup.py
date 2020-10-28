from sklearn import svm 
import pandas as pd

DATASET_PATH = "../../../datasets/titanic/"

class Dataset(object):

    def __init__(self):
        
        self.dataset_slice_1 = {
            "selected_feat" : "Survived, Pclass, Sex, Age, Fare",
            "dropped_feat"  : ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"]
        }
        
        self.training_set = pd.read_csv(DATASET_PATH + "train.csv", sep=",")
        self.__select_features()
    
    def __select_features(self):
        print(self.dataset_slice_1["dropped_feat"])
        self.training_set = self.training_set.drop(self.dataset_slice_1["dropped_feat"], axis=1)
        print(self.training_set)

class SVMModel(Dataset):
    def __init__(self):
        super().__init__()
        
    def train(self):
        pass

    def test():
        print(svm.__doc__)