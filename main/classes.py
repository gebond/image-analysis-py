from PIL import Image
import numpy as np
from time import gmtime, strftime


class Math:
    mask_0 = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]  # horizontal mask
    mask_1 = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]  # + 45 mask
    mask_2 = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]  # vertical mask
    mask_3 = [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]  # - 45 mask
    mask_4 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]  # dot of noise mask

    def __init__(self):
        pass

    def thresh_hold(self, t, x, y):
        if self.func_r(x, y) >= t:
            return 1
        else:
            return 0

    @staticmethod
    def create(rows, cols, img):

        matrix = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = (img.getpixel((i, j))[0] + img.getpixel((i, j))[1] + img.getpixel((i, j))[2]) / 3
        return matrix

    @staticmethod
    def normalize(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        max_val = -1000000
        min_val = 1000000
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] > max_val:
                    max_val = matrix[i][j]
                if matrix[i][j] < min_val:
                    min_val = matrix[i][j]

        norm_matrix = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                norm_matrix[i][j] = float(matrix[i][j] - min_val) / float(max_val - min_val)
        return norm_matrix

    @staticmethod
    def unnormalize(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        unnorm_matrix = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                unnorm_matrix[i][j] = float(matrix[i][j] * 255)
        resize = [[0 for x in range(rows)] for y in range(cols)]
        for i in range(cols):
            for j in range(rows):
                resize[i][j] = unnorm_matrix[j][i]
        return resize

    # apply horizontal 1d
    @staticmethod
    def apply1d(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_1d = [[0 for x in range(cols - 2)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols - 2):
                matrix_1d[i][j] = float(matrix[i][j + 1] - matrix[i][j])
        return matrix_1d

    # apply horizontal 2d
    @staticmethod
    def apply2d(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_2d = [[0 for x in range(cols - 2)] for y in range(rows)]
        for i in range(rows):
            for j in range(1, cols - 1):
                matrix_2d[i][j - 1] = matrix[i][j + 1] + matrix[i][j - 1] - 2 * matrix[i][j]
        return matrix_2d

    # apply horizontal 2d
    @staticmethod
    def func_r(matrix, i, j, mask_num):
        mask = [[]]
        if mask_num == 0:
            mask = Math.mask_0
        elif mask_num == 1:
            mask = Math.mask_1
        elif mask_num == 2:
            mask = Math.mask_2
        elif mask_num == 3:
            mask = Math.mask_3
        elif mask_num == 4:
            mask = Math.mask_4
        value = matrix[i - 1][j - 1] * mask[0][0] + matrix[i - 1][j] * mask[0][1] + matrix[i - 1][j + 1] * \
                mask[0][2] + matrix[i][j - 1] * mask[1][0] + matrix[i][j] * mask[1][1] + matrix[i][j + 1] * \
                mask[1][2] + matrix[i + 1][j - 1] * mask[2][0] + matrix[i + 1][j] * mask[2][1] + \
                matrix[i + 1][j + 1] * mask[2][2]
        return value

    # apply threshold t
    @staticmethod
    def apply_threshold(matrix, t, mask_num):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_filtered = [[0 for x in range(cols - 2)] for y in range(rows - 2)]
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if Math.func_r(matrix, i, j, mask_num) > t:
                    try:
                        matrix_filtered[i - 1][j - 1] = 1
                    except IndexError:
                        print('asd')
        return matrix_filtered

    # apply threshold t
    @staticmethod
    def apply_noise_threshold(matrix, t):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_filtered = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if abs(Math.func_r(matrix, i, j, 4)) > t:
                    matrix_filtered[i - 1][j - 1] = 0
                else:
                    matrix_filtered[i - 1][j - 1] = matrix[i][j]
        return matrix_filtered

    # apply threshold t
    @staticmethod
    def apply_threshold2(matrix, t1, t2):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_filtered = [[0 for x in range(cols - 2)] for y in range(rows - 2)]
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if Math.func_r(matrix, i, j, 0) > t1 or Math.func_r(matrix, i, j, 2) > t2:
                    matrix_filtered[i - 1][j - 1] = 1
        return matrix_filtered

    # apply laplasian
    @staticmethod
    def apply_laplasian(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_laplasian = [[0 for x in range(cols - 2)] for y in range(rows - 2)]
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                matrix_laplasian[i - 1][j - 1] = matrix[i][j + 1] + matrix[i][j - 1] + matrix[i + 1][j] + \
                                                 matrix[i - 1][j] - 4 * matrix[i][j]
        return matrix_laplasian

    # unnormalize, rotate and save matrix
    @staticmethod
    def save_matrix(num, type, matrix):
        unnorm_matrix = Math.unnormalize(matrix)
        img_array = np.asarray(unnorm_matrix)  # sample array
        im = Image.fromarray(img_array).convert('RGB')  # monochromatic image
        im.save('../output/car' + num + '-' + type + '.png')


class ImageBuilder:
    rows = 0
    cols = 0
    result = [[]]

    def __init__(self):
        pass

    # matrix from image
    def with_img_average(self, img):
        print("Image average color...")
        self.rows = img.size[0]
        self.cols = img.size[1]
        self.result = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                self.result[i][j] = float(
                    img.getpixel((i, j))[0] + img.getpixel((i, j))[1] + img.getpixel((i, j))[2]) / 3.0
        print("Done.")
        return self

    # matrix from image with channel
    def with_img_channel(self, img, channel):
        print("Image specific color " + channel + "...")
        channel_num = 0
        if channel == 'RED':
            channel_num = 0
        elif channel == 'GREEN':
            channel_num = 1
        elif channel == 'BLUE':
            channel_num = 2
        self.rows = img.size[0]
        self.cols = img.size[1]
        self.result = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                self.result[i][j] = float(img.getpixel((i, j))[channel_num])
        print("Done.")
        return self

    # normalize result matrix
    def normalize(self):
        print("Normalizing...")
        max_val = -1000000
        min_val = 1000000
        for i in range(self.rows):
            for j in range(self.cols):
                if self.result[i][j] > max_val:
                    max_val = self.result[i][j]
                if self.result[i][j] < min_val:
                    min_val = self.result[i][j]
        norm_matrix = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                norm_matrix[i][j] = float(self.result[i][j] - min_val) / float(max_val - min_val)
        self.result = norm_matrix
        print("Done.")
        return self

    def filter_only_positive(self):
        print("Filter positive values...")
        positive_matrix = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                if self.result[i][j] > 0:
                    positive_matrix[i][j] = float(self.result[i][j])
                else:
                    positive_matrix[i][j] = 0.0
        self.result = positive_matrix
        print("Done.")
        return self

    def laplasian(self):
        print("Calculating laplasian...")
        matrix_laplasian = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                matrix_laplasian[i - 1][j - 1] = float(
                    self.result[i][j + 1] + self.result[i][j - 1] + self.result[i + 1][j] + \
                    self.result[i - 1][j] - 4 * self.result[i][j])
        self.result = matrix_laplasian
        print("Done.")
        return self

    def horizontal_mask(self, t):
        print("Horizontal mask applying...")
        self.result = Math.apply_threshold(self.result, t, 2)
        print("Done.")
        return self

    def vertical_horizontal_mask(self, vert_t, hor_t):
        print("Vertical&Horizontal masks applying...")
        self.result = Math.apply_threshold2(self.result, vert_t, hor_t)
        print("Done.")
        return self

    def vertical_mask(self, t):
        print("Vertical mask applying...")
        self.result = Math.apply_threshold(self.result, t, 0)
        print("Done.")
        return self

    def sloop_45_mask(self, t):
        print("+45 sloop applying...")
        self.result = Math.apply_threshold(self.result, t, 3)
        print("Done.")
        return self

    def sloop_m_45_mask(self, t):
        print("-45 sloop applying...")
        self.result = Math.apply_threshold(self.result, t, 1)
        print("Done.")
        return self

    def noise_mask(self, t):
        print("Noise mask applying...")
        self.result = Math.apply_noise_threshold(self.result, t)
        print("Done.")
        return self

    def build(self):
        return self.result


class ImageUtils:

    def __init__(self):
        pass

    @staticmethod
    def save_matrix_as_image(num, type, matrix, un_norm=False):
        if un_norm:
            matrix = ImageUtils.un_normalize(matrix)
        matrix = ImageUtils.reflect(matrix)
        img_array = np.asarray(matrix)  # sample array
        im = Image.fromarray(img_array).convert('RGB')  # monochromatic image
        current = strftime("%H:%M:%S", gmtime())
        im.save('../output/' + current + '-' + num + '-' + type + '.png')

    @staticmethod
    def un_normalize(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        un_norm_matrix = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                un_norm_matrix[i][j] = float(matrix[i][j] * 255)
        return un_norm_matrix

    @staticmethod
    def reflect(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        reflected = [[0 for x in range(rows)] for y in range(cols)]
        for i in range(cols):
            for j in range(rows):
                reflected[i][j] = matrix[j][i]
        return reflected
