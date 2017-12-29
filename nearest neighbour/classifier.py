import numpy as np
import math
from operator import itemgetter
from collections import Counter

class Dataset:
    def __init__(self, path):
        self.K = 0
        self.D = 0
        self.obs = []
        self.file_path = path
        self.load_data_from_file()

    def load_data_from_file(self):
        lines = [line.rstrip('\n') for line in open(self.file_path)]
        self.K = int(lines[0])
        self.D = int(lines[1])
        matrix_order = int(math.sqrt(self.D))

        # fetch train_data from file
        index = 2
        num_lines = len(lines)
        while index < num_lines:
            class_id = int(lines[index])
            index += 1

            read_data = []
            for i, line in enumerate(lines[index: index + matrix_order]):
                read_data.extend([int(val) for val in line.split()])

            self.obs.append((read_data, class_id))
            index += matrix_order

def eucledian_distance(x, y):
    return np.linalg.norm(x - y)


class NN_Classifier:
    def __init__(self, training_Dataset):
        self.training_Dataset = training_Dataset

    def classify(self, input_vector, K=1):
        neighbours = self.get_nearest_neighbours(input_vector, K)
        return self.get_class_votes(neighbours)


    #returns list of tuples (distance, class index)
    def get_nearest_neighbours(self, input_vector, K):
        input_array = np.array(input_vector)
        distance_list = []

        for data in self.training_Dataset.obs:
            train_array = np.array(data[0])
            class_id = data[1]
            dist = eucledian_distance(input_array, train_array)
            distance_list.append((dist, class_id))

        sorted_list = sorted(distance_list, key=itemgetter(0))
        return sorted_list[:K]

    def get_class_votes(self, neighbour_distances):
        classes = [neighbour[1] for neighbour in neighbour_distances]
        count = Counter(classes)
        return count.most_common()[0][0]
