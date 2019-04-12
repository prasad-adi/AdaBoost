from mnist2image.mnist import load_mnist
import numpy as np
np.random.seed(42)

class HAAR:

    def __init__(self, number_of_rectangles):
        self.number_of_rectangles = number_of_rectangles

    def read_mnist(self):
        X, Y = load_mnist()
        main_indices = []
        for i in range(10):
            indices = np.where(Y == i)[0]
            size = int(len(indices) * 0.5)
            indices_20 = np.random.randint(0, len(indices), size=size)
            main_indices.extend(indices[indices_20])
        return X[main_indices], Y[main_indices]

    def __get_rectangle(self):
        rectangles = []
        while(len(rectangles) < self.number_of_rectangles):
            top_left = np.random.randint(0, 28, size = 2)
            bottom_right = np.random.randint(0, 28, size = 2)
            rectangle = Rectangle(top_left, bottom_right)
            if(top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1] and 130 <= rectangle.area <= 170):
                rectangles.append(rectangle)
        return rectangles

    def get_features(self, X):
        feature_array = np.zeros(200).reshape(1,200)
        for i in range(X.shape[0]):
            temp_array = self.get_datapoint(X[i])
            feature_array = np.append(feature_array, temp_array, axis = 0)

        return feature_array[1:,:]

    def get_datapoint(self, X):
        temp_array = np.array([])
        for j in range(len(self.rectangle)):
            temp_array = np.append(temp_array, self.rectangle[j].calculate_black_vertical(X))
            temp_array = np.append(temp_array, self.rectangle[j].calculate_black_horizontal(X))
        return temp_array.reshape(1, 200)

    def run(self):
        X, Y = self.read_mnist()
        self.rectangle = self.__get_rectangle()
        X = self.get_features(X)
        np.savetxt("./X.txt", X)
        np.savetxt(fname="./Y.txt", X =Y)


class Rectangle:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.width = abs(self.top_left[0] - self.bottom_right[0])
        self.height = abs(self.top_left[1] - self.bottom_right[1])
        self.area = self.width * self.height

    def calculate_black_vertical(self, X):
        vertical_break = int(self.width / 2)
        black_left = np.count_nonzero(X[:,:self.top_left[0] + vertical_break])
        black_right = np.count_nonzero(X[:,self.top_left[0] + vertical_break:])
        return black_left - black_right

    def calculate_black_horizontal(self, X):
        horizontal_break = int(self.height / 2)
        black_top = np.count_nonzero(X[:self.top_left[1] + horizontal_break, :])
        black_bottom = np.count_nonzero(X[self.top_left[1] + horizontal_break:, :])
        return  black_top - black_bottom


extract_haar = HAAR(number_of_rectangles = 100)
extract_haar.run()

