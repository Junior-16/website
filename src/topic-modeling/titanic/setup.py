from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
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

        self.label = "Survived"

        self.dataset_info = {
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

        self.__transform_gender_2continuos()
        self.__transform_embarked_2continuos()

    def __get_dropped_features(self):
        for feat in self.dataset.columns:
            if not feat in self.dataset_info["selected_feat"]:
                self.dataset_info["dropped_feat"].append(feat)

    def __select_features(self):
        self.dataset_info["slice"] = self.dataset.drop(self.dataset_info["dropped_feat"], axis=1)

    def __input_mean(self):
        means = {}
        for feat in self.dataset_info["input_mean_on"]:
            means[feat] = math.floor(self.dataset_info["slice"][feat].mean())

        for feat in means.keys():
            self.dataset_info["slice"][feat] = self.dataset_info["slice"][feat].fillna(means[feat])
        
        print(means)

    def __transform_gender_2continuos(self):
        self.dataset_info["slice"].loc[self.dataset_info["slice"]["Sex"] == "male", "Sex"] = 0
        self.dataset_info["slice"].loc[self.dataset_info["slice"]["Sex"] == "female", "Sex"] = 1

    def __transform_embarked_2continuos(self):
        # create new columns
        self.dataset_info["slice"]["S"] = 0
        self.dataset_info["slice"]["C"] = 0
        self.dataset_info["slice"]["Q"] = 0

        # fill with data
        self.dataset_info["slice"].loc[self.dataset_info["slice"]["Embarked"] == "S", "S"] = 1
        self.dataset_info["slice"].loc[self.dataset_info["slice"]["Embarked"] == "C", "C"] = 1
        self.dataset_info["slice"].loc[self.dataset_info["slice"]["Embarked"] == "Q", "Q"] = 1

        print(self.dataset_info["slice"])
        del self.dataset_info["slice"]["Embarked"]
        print(self.dataset_info["slice"])

    def get_labels(self):
        return self.dataset[self.label]

    def get_features(self):
        return self.dataset_info["slice"]

    def get_column(self, name):
        return self.dataset[name]

    def report(self):
        print("Amount of fetures: {}".format(self.dataset_info["feat_amount"]))
        print("Selected features: ")
        for feat in self.dataset_info["selected_feat"]: print("\t - {}".format(feat))
        print("Dropped features: ")
        for feat in self.dataset_info["dropped_feat"]: print("\t - {}".format(feat))

class Model:
    def __init__(self, train_dataset, test_dataset):
        self.training_set = train_dataset
        self.testing_set = test_dataset
        self.model = None

    def train(self):
        train_features = self.training_set.get_features()
        train_labels = self.training_set.get_labels()

        # print(train_features)
        # print(train_labels)

        # print("Features lenght: ", len(train_features))
        # print("Labels lenght: ", len(train_labels))

        self.model.fit(train_features, train_labels)

    def test(self):
        self.test_features = self.testing_set.get_features()
        self.predictions = self.model.predict(self.test_features)

    def write_output(self):
        output = open("../../../datasets/titanic/output.csv", "w")
        print("PassengerId,Survived", file=output)
        
        passengerId = self.testing_set.get_column("PassengerId")

        for i in range(len(passengerId)):
            print("{},{}".format(passengerId[i], self.predictions[i]), file=output)
        
        output.close()

class SVMModel(Model):

    '''
        TODO: Improve this shity description

        This abstraction uses Suport Vector Machine
        to model the dataset.
    '''

    def __init__(self, train_dataset, test_dataset):
        super().__init__(train_dataset, test_dataset)

        self.model = svm.SVC(kernel='linear')

        self.train()
        self.test()
        self.write_output()

class RandomForestModel(Model):
    def __init__(self, training_set, testing_set):
        super().__init__(training_set, testing_set)

        self.model = DecisionTreeClassifier()

        self.train()
        self.test()
        self.write_output()