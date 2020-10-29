from sklearn import svm, metrics
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

    def __init__(self, name, selected_feat, input_mean_on, label):

        self.label = label

        self.dataset_slice = {
            "slice"        : [],
            "feat_amount"  : len(selected_feat), 
            "input_mean_on": input_mean_on,
            "selected_feat": selected_feat,
            "dropped_feat" : [],
        }
        
        self.dataset = PD.read_csv(DATASET_PATH + name)

        self.__get_dropped_features()

        self.__select_features()
        self.__input_mean()

        self.__encode_gender()
    
    def __get_dropped_features(self):
        for feat in self.dataset.columns:
            if not feat in self.dataset_slice["selected_feat"]:
                self.dataset_slice["dropped_feat"].append(feat)

    def __select_features(self):
        self.dataset_slice["slice"] = self.dataset.drop(self.dataset_slice["dropped_feat"], axis=1)

    def __input_mean(self):
        means = {}
        for feat in self.dataset_slice["input_mean_on"]:
            means[feat] = math.floor(self.dataset_slice["slice"][feat].mean())

        for feat in means.keys():
            self.dataset_slice["slice"][feat] = self.dataset_slice["slice"][feat].fillna(means[feat])
        
        print(means)

    def __encode_gender(self):
        self.dataset_slice["slice"].loc[self.dataset_slice["slice"]["Sex"] == "male", "Sex"] = 0
        self.dataset_slice["slice"].loc[self.dataset_slice["slice"]["Sex"] == "female", "Sex"] = 1
        
    def get_labels(self):
        return self.dataset[self.label]

    def get_features(self):
        return self.dataset_slice["slice"]

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
        self.label = "Survived"
        self.features = ["Pclass", "Sex", "Age", "Fare"]
        self.input_mean_on = ["Age", "Fare"]
        self.training_set = Dataset("train.csv", self.features, self.input_mean_on, self.label)
        self.testing_set = Dataset("test.csv", self.features, self.input_mean_on, self.label)

        self.model = svm.SVC(kernel='linear')

        self.train()
        self.test()
        
    def train(self):
        train_features = self.training_set.get_features()
        train_labels = self.training_set.get_labels()

        print("Features lenght: ", len(train_features))
        print("Labels lenght: ", len(train_labels))

        self.model.fit(train_features, train_labels)

    def test(self):
        self.test_features = self.testing_set.get_features()
    
        self.predictions = self.model.predict(self.test_features)
        
    def write_output(self):
        pass



