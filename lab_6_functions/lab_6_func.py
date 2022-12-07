import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

NUMBER_OF_VECTOR_DIMENSIONS = 2
SAMPLE_SIZE_N = 200

X_LOWER_BORDER = -5
X_UPPER_BORDER = 5
Y_LOWER_BORDER = -4
Y_UPPER_BORDER = 4

# from lab 1
def generate_vector_X(A, M, n, N):
    left_border = 0
    right_border = 1
    m = (right_border + left_border) / 2
    number_of_realizations = 50
    Sn = np.zeros((n, N))
    for i in range(0, number_of_realizations, 1):
        Sn += np.random.uniform(left_border, right_border, (n, N)) - m
    standard_deviation = (right_border - left_border) / np.sqrt(12)
    E = Sn / (standard_deviation * np.sqrt(number_of_realizations))
    X = np.matmul(A, E) + np.reshape(M, (2, 1)) * np.ones((1, N))
    return X


def calculate_matrix_A(B):
    matrix_A = np.zeros((2, 2))
    matrix_A[0][0] = np.sqrt(B[0][0])
    matrix_A[0][1] = 0
    matrix_A[1][0] = B[0][1] / np.sqrt(B[0][0])
    matrix_A[1][1] = np.sqrt(B[1][1] - (B[0][1] ** 2) / B[0][0])
    return matrix_A


def calculate_mathematical_expectation_M(x):
    M = np.sum(x, axis=1) / SAMPLE_SIZE_N
    return M


def get_B_correlation_matrix_for_vector(x):
    M = calculate_mathematical_expectation_M(x)
    # M shape is (1, 2)
    B = np.zeros((2, 2))
    for i in range(0, SAMPLE_SIZE_N, 1):
        # sum for i xi * xi ^t  where x[:, i] = [ x, y ]^t  shape = (number of columns, number of rows) x.shape = (1,2)
        tmp = np.reshape(x[:, i], (2, 1))
        B += (np.matmul(tmp, np.transpose(tmp)))
    B /= SAMPLE_SIZE_N
    B -= np.matmul(np.reshape(M, (2, 1)), np.transpose(np.reshape(M, (2, 1))))
    return B


# from lab 2
def BayeslassificatorB(x, arrM, arrB, L):
    # create matrix of trurthfulness for dij and writing true, if d >= 0, else writing false
    # string with all true true will define class with it indexes
    X = np.reshape(x, (2, 1))
    table = np.eye(len(arrM)).astype(bool)

    flag = True
    for el in arrB:
        if el != arrB[0]:
            flag = False

    if flag==True:
        for i in range(0, len(arrM)-1):
            for j in range(i+1, len(arrM)):
                difM = np.reshape(arrM[i], (1, 2)) - np.reshape(arrM[j], (1, 2))
                sumM = np.reshape(arrM[i], (1, 2)) + np.reshape(arrM[j], (1, 2))
                tmp1 = np.matmul(np.matmul(difM, np.linalg.inv(arrB[0])), X)
                tmp2 = 0.5 * np.matmul(np.matmul(sumM, np.linalg.inv(arrB[0])), np.transpose(difM))
                dij = tmp1 - tmp2 + np.log(L)
                if dij >= 0:
                    table[i][j] = True
                    table[j][i] = False
                else:
                    table[i][j] = False
                    table[j][i] = True
        # print(table)
        for i in range(0, len(table)):
            if (table[i] == np.ones(len(table)).astype(bool)).all():
                return i
    else:
        for i in range(0, len(arrM)-1):
            for j in range(i+1, len(arrM)):
                divB = math.log(np.linalg.det(arrB[i]) / np.linalg.det(arrB[j]))
                dist1 = np.matmul(np.matmul(np.reshape(arrM[i], (1, 2)), np.linalg.inv(arrB[i])), np.reshape(arrM[i], (2, 1)))
                dist2 = np.matmul(np.matmul(np.reshape(arrM[j], (1, 2)), np.linalg.inv(arrB[j])), np.reshape(arrM[j], (2, 1)))
                tmp1 = np.matmul(np.matmul(np.transpose(X), (np.linalg.inv(arrB[j]) - np.linalg.inv(arrB[i]))), X)
                tmp2 = 2*(np.matmul(np.reshape(arrM[i], (1, 2)), np.linalg.inv(arrB[i])) -
                          np.matmul(np.reshape(arrM[j], (1, 2)), np.linalg.inv(arrB[j])))
                tmp2 = np.matmul(tmp2, X)
                tmp3 = divB + 2*np.log(L) - dist1 + dist2
                dij = tmp1 + tmp2 + tmp3
                if dij >= 0:
                    table[i][j] = True
                    table[j][i] = False
                else:
                    table[i][j] = False
                    table[j][i] = True
        # print(table)
        for i in range(0, len(table)):
            if (table[i] == np.ones(len(table)).astype(bool)).all():
                return i


# from current lab

def Parzen_classificator(x, train_classes, train_B):
    n_classes = len(train_classes)

    f = np.zeros((n_classes, ))
    P = np.zeros((n_classes, ))

    cnt = 0
    k = 0.25
    for i in range(0, n_classes):
        n_vectors = len(np.transpose(train_classes[i]))
        h = n_vectors**(-k / NUMBER_OF_VECTOR_DIMENSIONS)
        sum = 0
        const = pow((2.0 * np.pi), (-NUMBER_OF_VECTOR_DIMENSIONS / 2.0)) * pow(h, -NUMBER_OF_VECTOR_DIMENSIONS) *\
                pow(np.linalg.det(train_B[i]), -0.5)
        for x_i in np.transpose(train_classes[i]):
            dist = np.matmul(np.matmul((x-x_i), np.linalg.inv(train_B[i])), (x-x_i))
            power = -0.5 * pow(h, -2) * dist
            sum += const * np.exp(power)
        sum /= n_vectors
        cnt += n_vectors

        f[i] = sum
        P[i] = n_vectors
    P /= cnt
    return np.argmax(P*f)


def d(x, z):
    dist = np.sum(np.square(x-z))
    return np.sqrt(dist)


Distance = np.vectorize(d, signature='(n),(m)->()')  # takes one axis array from first arg and from second arg
# output is number  for each one axis array in first argument

def K_neighbors_classificator(x, train_classes, K):
    all_vectors = np.transpose(np.concatenate(train_classes, axis=1))
    r = np.zeros((len(np.transpose(train_classes[0])), ))

    for i in range(1, len(train_classes)):
        size = np.shape(train_classes[i])
        r = np.concatenate([r, np.ones((size[1], ))*i])

    distances = Distance(all_vectors, x)
    neighbors = distances.argpartition(K)[:K]
    neighbors_classes = list(r[neighbors])
    num_class = max(set(neighbors_classes), key=neighbors_classes.count)
    return int(num_class)


def get_classes(test_X, train_X, func, **kwargs):
    classes = []
    for i in range(0, len(train_X)):
        classes.append(np.empty((NUMBER_OF_VECTOR_DIMENSIONS, 1)))

    for el in np.transpose(np.concatenate(test_X, axis=1)):
        num_class = -1
        if func == Parzen_classificator:
            arr_B = kwargs["B"]
            num_class = func(el, train_X, arr_B)  # number of classes starts from 0
        elif func == K_neighbors_classificator:
            K = kwargs["K"]
            num_class = func(el, train_X, K)  # number of classes starts from 0
        else:
            arr_M = kwargs["M"]
            arr_B = kwargs["B"]
            num_class = BayeslassificatorB(el, arr_M, arr_B, 1)  # number of classes starts from 0
        classes[num_class] = np.concatenate([classes[num_class], np.reshape(el, (NUMBER_OF_VECTOR_DIMENSIONS, 1))], axis=1)

    for i in range(0, len(train_X)):
        classes[i] = classes[i][:, 1:]
    return classes


def view_classes(true_classes, title, *args):
    if len(args) > 0:
        fig = plt.figure(figsize=(16, 7))
        plt.suptitle(title)
        fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(title)

    plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
    plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)
    c = ['r+', 'g+', 'b+']
    for i in range(0, len(true_classes)):
        plt.plot(true_classes[i][0], true_classes[i][1], c[i], label=f"class {i}")
    plt.legend()

    if len(args) > 0:
        classes = args[0]

        fig.add_subplot(122)
        plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
        plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)
        c = ['r+', 'g+', 'b+']
        for i in range(0, len(classes)):
            plt.plot(classes[i][0], classes[i][1], c[i], label=f"class {i}")

        errors = calc_errors(classes, true_classes)
        errors = np.transpose(errors)
        plt.scatter(errors[0], errors[1], s=50, linewidth=1, facecolors='none',
                    edgecolors='purple', label="Wrong classified vectors", alpha=0.7)
    plt.legend()



def calc_errors(classes, true_classes):
    errors = []
    for i in range(0, len(classes)):  # in every class
        for el in np.transpose(classes[i]):  # moving by vectors
            if not (el == np.transpose(true_classes[i])).all(axis=1).any():  # check presence in true class
                errors.append(el)
    return errors