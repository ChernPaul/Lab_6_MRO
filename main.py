import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show
from lab_6_functions.lab_6_func import calculate_matrix_A, generate_vector_X, \
    get_classes, calc_errors, view_classes, Parzen_classificator, K_neighbors_classificator

TASK_VARIANT = 20
VALUE_ONE_HUNDRED = 100

M1 = [1, -1]
M2 = [-2, -2]
M3 = [-1, 1]

B1 = [[0.5, 0.0],
      [0.0, 0.5]]

B2 = [[0.4, 0.1],
      [0.1, 0.6]]

B3 = [[0.6, -0.2],
      [-0.2, 0.6]]


NUMBER_OF_VECTOR_DIMENSIONS = 2
SAMPLE_SIZE_N = 200

PROBABILITY_HALF_OF_ONE = 0.5
PROBABILITY_ONE_OF_TREE = float(1/3)

C_MATRIX_OF_FINE = np.array([[0.0, 1.0],
                             [1.0, 0.0]])


if __name__ == '__main__':

    # 1. Синтезировать дополнительные реализации для каждого из классов (использовать
    # исполняемый файл из л.р. № 1).

    """
    A1 = calculate_matrix_A(B1)
    A2 = calculate_matrix_A(B2)
    A3 = calculate_matrix_A(B3)

    vector_1 = generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_2 = generate_vector_X(A1, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_3 = generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_4 = generate_vector_X(A2, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_5 = generate_vector_X(A3, M3, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)

    np.save("saves\\vector_1", vector_1)
    np.save("saves\\vector_2", vector_2)
    np.save("saves\\vector_3", vector_3)
    np.save("saves\\vector_4", vector_4)
    np.save("saves\\vector_5", vector_5)
   

    

    

    
    A1 = calculate_matrix_A(B1)
    A2 = calculate_matrix_A(B2)
    A3 = calculate_matrix_A(B3)
   
    vector_1_new = generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_2_new = generate_vector_X(A1, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_3_new = generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_4_new = generate_vector_X(A2, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_5_new = generate_vector_X(A3, M3, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)

    np.save("saves\\vector_1_new", vector_1_new)
    np.save("saves\\vector_2_new", vector_2_new)
    np.save("saves\\vector_3_new", vector_3_new)
    np.save("saves\\vector_4_new", vector_4_new)
    np.save("saves\\vector_5_new", vector_5_new)
    """

    vector_1 = np.load("saves\\vector_1.npy")
    vector_2 = np.load("saves\\vector_2.npy")
    vector_3 = np.load("saves\\vector_3.npy")
    vector_4 = np.load("saves\\vector_4.npy")
    vector_5 = np.load("saves\\vector_5.npy")

    vector_1_new = np.load("saves\\vector_1_new.npy")
    vector_2_new = np.load("saves\\vector_2_new.npy")
    vector_3_new = np.load("saves\\vector_3_new.npy")
    vector_4_new = np.load("saves\\vector_4_new.npy")
    vector_5_new = np.load("saves\\vector_5_new.npy")


    fig_new = plt.figure(figsize=(10, 10))
    # fig_new.add_subplot(2, 2, 1)
    # plt.title("Train data for 2 classes equal B")
    # plt.plot(vector_1_new[0], vector_1_new[1], 'r+')
    # plt.plot(vector_2_new[0], vector_2_new[1], 'b+')

    fig_new.add_subplot(1, 2, 1)
    plt.title("Train data for 3 classes unequal B")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(vector_3_new[0], vector_3_new[1], 'r+')
    plt.plot(vector_4_new[0], vector_4_new[1], 'g+')
    plt.plot(vector_5_new[0], vector_5_new[1], 'b+')

    # fig_new.add_subplot(2, 2, 3)
    # plt.title("Test data for 2 classes equal B")
    # plt.plot(vector_1[0], vector_1[1], 'r+')
    # plt.plot(vector_2[0], vector_2[1], 'b+')


    fig_new.add_subplot(1, 2, 2)
    plt.title("Test data for 3 classes unequal B")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(vector_3[0], vector_3[1], 'r+')
    plt.plot(vector_4[0], vector_4[1], 'g+')
    plt.plot(vector_5[0], vector_5[1], 'b+')

    show()

    # 2. Построить классификатор, основанный на непараметрической оценки Парзена,
    # используя сгенерированные в п.1 данные как обучающие выборки, а данные из
    # первой лабораторной работы - как тестовые. В качестве ядра взять гауссовское
    # (10), величину h взять в виде (11). Оценить эмпирический риск - оценку
    # суммарной вероятности ошибочной классификации

    train = [vector_3_new, vector_4_new, vector_5_new]
    test = [vector_3, vector_4, vector_5]
    M = [M1, M2, M3]
    B = [B1, B2, B3]

    Parzen_classes = get_classes(test, train, Parzen_classificator, B=B)
    Parzen_errors = calc_errors(Parzen_classes, test)
    wrong_prob_value = len(Parzen_errors) / (len(test) * SAMPLE_SIZE_N)

    print(f"Summary probability of wrong classification for Parsen: ", wrong_prob_value)

    view_classes(train, "Train Classes ")
    view_classes(test, "Parsen classification result", Parzen_classes)


    # 3. Построить классификатор, основанный на методе К ближайших соседей (для
    # K=1,3,5), используя сгенерированные в п.1 данные как обучающие выборки, а
    # данные из первой лабораторной работы - как тестовые. Оценить эмпирический
    # риск - оценку суммарной вероятности ошибочной классификации.

    # 4. Сравнить полученные в пп.2-3 классификаторы и качество их работы с
    # байесовским классификатором из л.р.№2.
    for k in range(1, 6, 2):
        K_neighbors_classes = get_classes(test, train, K_neighbors_classificator, K=k)
        K_neighbors_errors = calc_errors(K_neighbors_classes, test)
        wrong_prob_value = len(K_neighbors_errors) / (len(test) * SAMPLE_SIZE_N)
        print(f"Summary probability of wrong classification for {k} closest neighbours: ", wrong_prob_value)
        view_classes(test, f"Classification for {k} closest neighbours", K_neighbors_classes)

    Bayes_classes = get_classes(test, train, func="Bayes", M=M, B=B)
    Bayes_errors = calc_errors(Bayes_classes, test)
    wrong_prob_value = len(Bayes_errors) / (len(test) * SAMPLE_SIZE_N)
    print(f"Summary probability of wrong classification for Bayes: ", wrong_prob_value)
    view_classes(test, f"Bayes classificator", Bayes_classes)
    show()





