from classifier import *
import sys

def evaluate_dataset(trainDataset, testDataset, k_range):

    classifier = NN_Classifier(trainDataset)
    data_len = len(testDataset.obs)

    for k in k_range:
        print("\nEvaluating data for K = {}".format(k))
        error_count = 0

        for i, data in enumerate(testDataset.obs):
            if data[1] != classifier.classify(data[0], k):
                error_count += 1
            sys.stdout.write("\rComplete : {:0.2%}".format((i+1) / data_len))
            sys.stdout.flush()

        print("\nError rate for K = {} is {:%}".format(k, error_count / data_len))

if __name__ == "__main__":

    print("\nLoading usps training data ")
    train_Dataset = Dataset("usps.train")

    print("\nLoading usps test data ")
    test_Dataset = Dataset("usps.test")

    print("\nEvaluating usps test data:")
    evaluate_dataset(train_Dataset, test_Dataset, k_range=range(1, 11))

    print("\nEvaluating usps training data:")
    evaluate_dataset(train_Dataset, train_Dataset, k_range=range(1, 11))
