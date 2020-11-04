from setup import *

def run_svm(dataset):    
    svm_model = SVMModel(dataset)

def run_decision_tree(dataset):
    dt = DecisionTreeModel(dataset)

def run_random_forest(dataset):
    rf = RandomForestModel(dataset)

if __name__ == "__main__":
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
    dataset = Dataset(features)
    # print("Training set")
    # print(dataset.get_train_labels())
    # print(dataset.get_train_features())

    # print("Testing set")
    # print(dataset.get_test_features())
    # run_svm(dataset)
    # run_decision_tree(dataset)
    run_random_forest(dataset)