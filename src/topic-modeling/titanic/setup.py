from sklearn import svm 
import pandas as PD
import math

DATASET_PATH = "../../../datasets/titanic/"

class Dataset:

    '''
        TODO: Improve this shity description

        Dataset class that performs feature selection
        and mean inputation. Constructor accepts dataset
        name, list of features present in the dataset (the missing ones
        will be dropped), and features where the null values will be
        replace by the feature mean.

        @author Junior Vitor Ramisch <junior.ramisch@gmail.com> 
    '''

    def __init__(self, name, selected_feat, input_mean_on):
        
        self.dataset_slice = {
            "slice"        : [],
            "feat_amount"  : len(selected_feat), 
            "input_mean_on": input_mean_on,
            "selected_feat": selected_feat,
            "dropped_feat" : [],
        }
        
        self.training_set = PD.read_csv(DATASET_PATH + name)

        self.__get_dropped_features()

        self.__select_features()
        self.__input_mean()
    
    def __get_dropped_features(self):
        for feat in self.training_set.columns:
            if not feat in self.dataset_slice["selected_feat"]:
                self.dataset_slice["dropped_feat"].append(feat)

    def __select_features(self):
        self.dataset_slice["slice"] = self.training_set.drop(self.dataset_slice["dropped_feat"], axis=1)

    def __input_mean(self):
        means = {}
        for feat in self.dataset_slice["input_mean_on"]:
            means[feat] = math.floor(self.dataset_slice["slice"][feat].mean())

        for feat in means.keys():
            self.dataset_slice["slice"][feat] = self.dataset_slice["slice"][feat].fillna(means[feat])

    def report(self):
        print("Amount of fetures: {}".format(self.dataset_slice["feat_amount"]))
        print("Selected features: ")
        for feat in self.dataset_slice["selected_feat"]: print("\t - {}".format(feat))
        print("Dropped features: ")
        for feat in self.dataset_slice["dropped_feat"]: print("\t - {}".format(feat))


class SVMModel:

    '''
        TODO: Improve this shity description

        This abstraction uses Suport Vector Machine
        to model the datasets.
    '''

    def __init__(self):
        self.features = ["Survived", "Pclass", "Sex", "Age", "Fare"]
        self.input_mean_on = ["Age"]
        self.training_set = Dataset("train.csv", self.features, self.input_mean_on)
        self.training_set.report()
        
    def train(self):
        pass

    def test():
        print(svm.__doc__)
