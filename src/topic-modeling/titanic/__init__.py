from setup import *

def run_svm(training_set, testing_set):    
    svm_model = SVMModel(training_set, testing_set)

def run_random_forest(training_set, testing_set):
    rf = RandomForestModel(training_set, testing_set)

if __name__ == "__main__":
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
    input_mean_on = ["Age", "Fare"]
    training_set = Dataset("train.csv", features, input_mean_on)
    testing_set = Dataset("test.csv", features, input_mean_on)
    
    # run_svm(training_set, testing_set)
    run_random_forest(training_set, testing_set)