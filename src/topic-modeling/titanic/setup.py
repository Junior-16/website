from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as PD
import numpy as np
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

    def __init__(self, selected_feat):

        self.label = "Survived"
       
        self.selected_feat = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
        self.dropped_feat = []

        self.train = PD.read_csv(DATASET_PATH + "train.csv")
        self.test = PD.read_csv(DATASET_PATH + "test.csv") 
        self.datasets = [self.train, self.test]

        self.__get_dropped_features()

        self.__input_mean_on_age()
        self.__input_median_on_fare()

        self.__transform_gender_2continuos()
        self.__transform_embarked_2continuos()
        self.__create_features()

        self.__select_features()

    def __get_dropped_features(self):
        for feat in self.train.columns:
            if not feat in self.selected_feat:
                self.dropped_feat.append(feat)

    def __select_features(self):
        self.train = self.train.drop(self.dropped_feat, axis=1)
        self.dropped_feat.remove("FareBand")
        self.dropped_feat.remove("Survived")
        self.test = self.test.drop(self.dropped_feat, axis=1)

    def __input_mean_on_age(self): # DONE
        
        for dataset in self.datasets:
            age_avg = dataset['Age'].mean()
            age_std = dataset['Age'].std()
            age_null_count = dataset['Age'].isnull().sum()
            
            age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
            dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
            dataset['Age'] = dataset['Age'].astype(int)
    
        for dataset in self.datasets:
            dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    def __input_median_on_fare(self): # DONE

        for dataset in self.datasets:
            dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

        self.train["FareBand"] = PD.qcut(self.train['Fare'], 4)
        self.dropped_feat.append("FareBand")

        for dataset in self.datasets:
            dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
            dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
            dataset['Fare'] = dataset['Fare'].astype(int)

    def __transform_gender_2continuos(self):
        for dataset in self.datasets:
            dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    def __transform_embarked_2continuos(self): # DONE

        for dataset in self.datasets:
            dataset["Embarked"] = dataset["Embarked"].fillna("S")

        for dataset in self.datasets:
            dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # By kaggle notebook (I was briefed)
    def __create_features(self):
        for dataset in self.datasets:
            dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

        for dataset in self.datasets:
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        for dataset in self.datasets:
            dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')

        for dataset in self.datasets:
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
        for dataset in self.datasets:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)

    def get_train_labels(self):
        return self.datasets[0][self.label]

    def get_train_features(self):
        return self.train

    def get_test_features(self):
        return self.test

    def get_column_from_test(self, column):
        return self.datasets[1][column]

    def get_column_from_train(self, column):
        return self.datasets[0][column]

    def report_train(self):
        print(self.train)

class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_labels = dataset.get_train_labels()
        self.train_features = dataset.get_train_features()
        self.test_features = dataset.get_test_features()
        self.model = None

    def train(self):
        self.model.fit(self.train_features, self.train_labels)

    def test(self):
        self.predictions = self.model.predict(self.test_features)

    def write_output(self, modelType):
        output = open("../../../datasets/titanic/output-{}.csv".format(modelType), "w")
        print("PassengerId,Survived", file=output)
        
        passengerId = self.dataset.get_column_from_test("PassengerId")

        for i in range(len(passengerId)):
            print("{},{}".format(passengerId[i], self.predictions[i]), file=output)
        
        output.close()

class SVMModel(Model):

    '''
        TODO: Improve this shity description

        This abstraction uses Suport Vector Machine
        to model the dataset.
    '''

    def __init__(self, dataset):
        super().__init__(dataset)

        self.model = svm.SVC(kernel='linear')

        self.train()
        self.test()
        self.write_output("svm")

class DecisionTreeModel(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.model = DecisionTreeClassifier()

        self.train()
        self.test()
        self.write_output("decision_tree")

class RandomForestModel(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.model = RandomForestClassifier(n_estimators=100)

        self.train()
        self.test()
        self.write_output("random_forest")