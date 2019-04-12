import numpy as np



class HaarFeatures:

    def __init__(self, n_rectangles):
        self.n_rectangles = n_rectangles    # the number of rectangles to extract
        self.rectangles = []        # the list of rectangles

    def get_random_rectangles(self):
        """
        Generate the random coordinates of given number "n_rectangles" of rectangles
        on the 28x28 image.
        """
        while len(self.rectangles) < self.n_rectangles:
            upper_left = [np.random.randint(0, 28) for i in range(2)]   # upper-left corner coordinate
            lower_right = [np.random.randint(0, 28) for i in range(2)]  # lower-right corner coordinate
            # Have upper left corner less than lower right corner of the rectangle
            if upper_left[0] < lower_right[0] and upper_left[1] < lower_right[1]:
                currentRect = Rectangle(upper_left, lower_right)
                currentArea = currentRect.area()
                # Only keep the rectangles whose area is 130 to 170
                if 130 <= currentArea <= 170:
                    self.rectangles.append(currentRect)
                    #print("Upper Left ", upper_left, " Lower right ", lower_right, " Area: ", currentRect.area())

    def black(self, x):
        """
        Calculate the number of black pixels for all coordinate [i,j] of the image
        relative to the (0,0) upper-left of the image.
        :param x: the image.
        """
        n = x.shape[0]
        count_arr = np.array([[0] * 28] * 28)   # array to keep the count for all [i,j] coordinates
        # Base: i = j = 0, count_arr[i, j] = 0
        for i in range(1, n):
            for j in range(1, n):
                if x[i, j] > 0:
                    count_arr[i, j] = count_arr[i, j - 1] + count_arr[i - 1, j] - count_arr[i - 1, j - 1] + 1
                else:
                    count_arr[i, j] = count_arr[i, j - 1] + count_arr[i - 1, j] - count_arr[i - 1, j - 1]
        self.count_black = count_arr

    def calculate_black_rectangle(self, rectangle):
        """
        Calculate the number of black pixels in the given rectangle of the image.
        :param rectangle: the Rectangle object.
        :return: the number of black pixels in the Rectangle.
        """
        result = self.count_black[rectangle.lower_right[0], rectangle.lower_right[1]] - \
                self.count_black[rectangle.upper_right[0], rectangle.upper_right[1]] - \
                self.count_black[rectangle.lower_left[0], rectangle.lower_left[1]] + \
                self.count_black[rectangle.upper_left[0], rectangle.upper_left[1]]
        return result

    def getFeatures(self, X):
        """
        Calculate the vertical and horizontal feature of the Image.
        :param X: the input X.
        :return: the features.
        """
        n = X.shape[0]
        self.get_random_rectangles()
        features = []
        for i in range(n):
            # Reshape the image into a 2D array
            x_i = X[i, :].reshape(28, 28)
            # Calculate the number of black pixels at all rectangles whose upper left corner is origin
            self.black(x_i)
            # Get the Image Rectangles
            img_features = []     # to keep the rectangles of the Image
            for r in self.rectangles:
                # print("Rect: ", r)
                # Extract top and bottom vertical rectangles and left and right horizontal rectangles.
                top_vert_rect = Rectangle(r.upper_left, (r.upper_right[0], r.upper_right[1] + (r.height // 2)))
                bottom_vert_rect = Rectangle([r.upper_left[0], r.upper_left[1] + (r.height // 2)], r.lower_right)
                left_hori_rect = Rectangle(r.upper_left, (r.lower_left[0] + (r.width // 2), r.lower_left[1]))
                right_hori_rect = Rectangle([r.upper_left[0] + (r.width // 2) , r.upper_left[1]], r.lower_right)
                # print("Top: ", top_vert_rect)
                # print("Botton: ", bottom_vert_rect)
                # print("Left: ", left_hori_rect)
                # print("Right: ", right_hori_rect)

                # Calculate the number of black pixels in each rectangle
                black_top_vert_rect = self.calculate_black_rectangle(top_vert_rect)
                black_bottom_vert_rect = self.calculate_black_rectangle(bottom_vert_rect)
                black_left_hori_rect = self.calculate_black_rectangle(left_hori_rect)
                black_right_hori_rect = self.calculate_black_rectangle(right_hori_rect)

                # Calculate vertial and horizontal feature value of each rectangle
                vertical_feature_val = black_top_vert_rect - black_bottom_vert_rect
                horizontal_feature_val = black_left_hori_rect - black_right_hori_rect

                # print("Top: ", black_top_vert_rect, " Bottom: ", black_bottom_vert_rect,
                #       " Left: ", black_left_hori_rect, " Right: ", black_right_hori_rect)
                # print("Vertical Feature: ", vertical_feature_val, " Horizontal Feature: ", horizontal_feature_val)

                img_features.append(vertical_feature_val)
                img_features.append(horizontal_feature_val)
            features.append(img_features)
        return np.array(features)

"""
Represents a Rectangle.
"""
class Rectangle:

    def __init__(self, upper_left, lower_right):
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.lower_left = [self.upper_left[0], self.lower_right[1]]
        self.upper_right = [self.lower_right[0], self.upper_left[1]]
        self.width = self.lower_right[0] - self.lower_left[0]
        self.height =  self.lower_left[1] - self.upper_left[1]

    def area(self):
        return self.width * self.height

    def __str__(self):
        return "A: " + str(self.upper_left) + " B: " + str(self.upper_right) + \
               " C: " + str(self.lower_left) + " D: " + str(self.lower_right) + \
                " width: " + str(self.width) + " height: " + str(self.height)
