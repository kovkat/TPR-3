import numpy as np


def print_matrix(matrix, column_order= None):

    if column_order:
        # Формування заголовків на основі переданих індексів
        headers = [f"K{col}" for col in column_order]
    else:
        # Використання стандартних заголовків
        headers = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12"]
    
    print("Index  " + " ".join(f"{h:>5}" for h in headers))  # Вивід заголовків
    print("-" * (7 + len(headers) * 6))
    for idx, row in enumerate(matrix, start=1):
        print(f"{idx:<5} | " + " ".join(f"{x:>5}" for x in row))

def print_matrix_pod(matrix):
        # Використання стандартних заголовків
    headers = ["𝜓1", "𝜓2", "𝜓3", "𝜓4", "𝜓5", "𝜓6", "𝜓7", "𝜓8", "𝜓9", "𝜓10", "𝜓11", "𝜓12"]
    
    print("Index  " + " ".join(f"{h:>5}" for h in headers))  # Вивід заголовків
    print("-" * (7 + len(headers) * 6))
    for idx, row in enumerate(matrix, start=1):
        print(f"{idx:<5} | " + " ".join(f"{x:>5}" for x in row))

def calculate_delta(x, y):
    return [xi - yi for xi, yi in zip(x, y)]

# Функція сигнум
def calculate_sigma(delta):
    return list(map(np.sign, delta))

# Створення матриці векторів знаків різниць і обчислення дельт
def calculate_sigma_table(matrix):
    n = len(matrix)
    sigma_table = []
    delta_table = []
    
    for i in range(n):
        sigma_row = []
        delta_row = []
        for j in range(n):
            delta = calculate_delta(matrix[i], matrix[j])
            sigma = calculate_sigma(delta)
            delta_row.append(delta)
            sigma_row.append(sigma)
        delta_table.append(delta_row)
        sigma_table.append(sigma_row)
    
    return delta_table, sigma_table

#ВІДНОШЕННЯ ПАРЕТО
def pareto(matrix):
    print("\nРозв'язок Парето:")
    delta_table, sigma_table = calculate_sigma_table(matrix)
    
    # Вивід дельта і сигма 
    n = len(matrix)

    # print("\nВектори Δ :")
    # for i in range(n):
    #     for j in range(n):
    #         print(f"Δ({i+1}, {j+1}): {delta_table[i][j]}")
    
    # print("\nВектори σ:")
    # for i in range(n):
    #     for j in range(n):
    #         print(f"σ({i+1}, {j+1}): {sigma_table[i][j]}")
    
    # Побудова матриці r0 (Парето-відношення)
    R0 = [
        [0 if any(s < 0 for s in sigma_table[i][j]) else 1 for j in range(len(matrix))]
        for i in range(len(matrix))
    ]
    
    # Форматований вивід матриці r0
    print("\nR0:")
    for row in R0:
        print(" ".join(map(str, row)))
    
    return R0
#МАЖОРИТАРНЕ ВІДНОШЕННЯ
def mag(matrix):
    delta_table, sigma_table = calculate_sigma_table(matrix)
    n = len(matrix)
    print("\nМажоритарне відношення:")
    # обчислюємо суму кожного вектора 
    sum_vector = np.sum(sigma_table, axis=2)

    # print("\nВектори суми σ:")
    # for i in range(n):
    #     for j in range(n):
    #         print(f"σ({i+1}, {j+1}): {sum_vector[i][j]}")


    Rm = np.where(sum_vector > 0, 1, 0)
     # Форматований вивід матриці r0
    print("\nRm:")
    for row in Rm:
        print(" ".join(map(str, row)))
    
    return Rm
#ЛЕКСИКОГРАФІЧНЕ ВІДНОШЕННЯ
def lek(matrix):
    print("\nЛексикографічне відношення:")
    delta_table, sigma_table = calculate_sigma_table(matrix)
    
    # порядок впорядкування k3 > k10 > k4 > k8 > k2 > k6 > k12 > k5 > k7 > k9 > k1 > k11
    column_order = [3, 10, 4, 8, 2, 6, 12, 5, 7, 9, 1, 11]
    
    # Перестановка стовпців відповідно до нового порядку
    sorted_matrix = np.array(matrix)[:, [i - 1 for i in column_order]]
    print("\nВпорядкована матриця:")
    print_matrix(sorted_matrix, column_order)
    
    # Побудова матриці лексикографічного відношення
    Rl = []
    for i in range(len(sorted_matrix)):
        Rl.append([])
        for j in range(len(sorted_matrix)):
            if i == j:
                Rl[i].append(0)  # Сам до себе відношення відсутнє
                continue

            # Перевірка відношення за впорядкованими критеріями
            for k in range(sorted_matrix.shape[1]):
                if sigma_table[i][j][column_order[k] - 1] > 0:  # `i` переважає `j`
                    Rl[i].append(1)
                    break
                elif sigma_table[i][j][column_order[k] - 1] < 0:  # `j` переважає `i`
                    Rl[i].append(0)
                    break
            else:
                # Якщо після всіх критеріїв відношення не знайдено, вважати рівними
                Rl[i].append(0)
    
    print("\nRl:")
    for row in Rl:
        print(" ".join(map(str, row)))
    return Rl

def compare_elements(m):
    # Порівняння елементів для створення матриць
    n = len(m)

    i_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    n_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j and i_matrix[i][j]==i_matrix[j][i]== 1:
                i_matrix[i][j] = 1
                continue
            if m[i][j] < m[j][i]:
                p_matrix[j][i] = 1
            if m[i][j] > m[j][i]:
                p_matrix[i][j] = 1
            if m[i][j] == m[j][i] == 0:
                n_matrix[i][j] = 1
                n_matrix[j][i] = 1
            if m[i][j] == m[j][i] == 1:
                i_matrix[i][j] = 1
                i_matrix[j][i] = 1

    i_matrix = i_matrix.astype(int)
    p_matrix = p_matrix.astype(int)
    n_matrix = n_matrix.astype(int)
    return i_matrix, p_matrix, n_matrix


def process_class(matrix, class_label):
    i_matrix, p_matrix, n_matrix = compare_elements(matrix)
    

    print( f"\nСиметрична частина I0 для {class_label}")
    for row in i_matrix:
        print(" ".join(map(str, row)))
    
    print(f"\nАсиметрична частина P0 для {class_label}")
    for row in p_matrix:
        print(" ".join(map(str, row)))
    print(f"\nНепорівнюваності частина N0 для {class_label}")
    for row in n_matrix:
        print(" ".join(map(str, row)))

#ВІДНОШЕННЯ БЕРЕЗОВСЬКОГО
def ber (matrix):
    print("\nВідношення Березовського")
    n = len(matrix)
    k1 = [1, 4, 6]
    k2 = [7, 11]
    k3 = [2, 3, 5, 8, 9, 10, 12]
    
    k1_matrix = np.array(matrix)[:, np.array(k1) - 1]
    k2_matrix = np.array(matrix)[:, np.array(k2) - 1]
    k3_matrix = np.array(matrix)[:, np.array(k3) - 1]
    
    print("\n Клас KI")
    print_matrix(k1_matrix, k1)
    k1_pareto = pareto(k1_matrix)
    process_class(k1_pareto, "KI")
    i_matrix_1, p_matrix_1, n_matrix_1 = compare_elements(k1_pareto)

    print("\n Клас KII")
    print_matrix(k2_matrix, k2)
    k2_pareto = pareto(k2_matrix)
    process_class(k2_pareto, "KII")
    i_matrix_2, p_matrix_2, n_matrix_2 = compare_elements(k2_pareto)

    print("\n Клас KIII")
    print_matrix(k3_matrix, k3)
    k3_pareto = pareto(k3_matrix)
    process_class(k3_pareto, "KIII")
    i_matrix_3, p_matrix_3, n_matrix_3 = compare_elements(k3_pareto)
    rb1 = np.zeros((n, n))
    rb1 =rb1.astype(int)
    #Cкладаємо першу частину Березовського 
    for i in range(n):
        for j in range(n):
            if p_matrix_2[i][j] == p_matrix_1[i][j] == 1:
                rb1[i][j] = 1
            if p_matrix_2[i][j] == n_matrix_1[i][j] == 1:
                rb1[i][j] = 1
            if p_matrix_2[i][j] == i_matrix_1[i][j] == 1:
                rb1[i][j] = 1
            if i_matrix_2[i][j] == p_matrix_1[i][j] == 1:
                rb1[i][j] = 1
    print("\nПерша частина.Порівняння класів KII та KI")
    for row in rb1:
        print(" ".join(map(str, row)))
    Rb =np.zeros((n, n))
    Rb =Rb.astype(int)
    i_matrix_rb1, p_matrix_rb1, n_matrix_rb1 = compare_elements(rb1)
    #Cкладаємо першу частину Березовського 
    for i in range(n):
        for j in range(n):
            if p_matrix_3[i][j] == p_matrix_rb1[i][j] == 1:
                Rb[i][j] = 1
            if p_matrix_3[i][j] == n_matrix_rb1[i][j] == 1:
                Rb[i][j] = 1
            if p_matrix_3[i][j] == i_matrix_rb1[i][j] == 1:
                Rb[i][j] = 1
            if i_matrix_3[i][j] == p_matrix_rb1[i][j] == 1:
                Rb[i][j] = 1
    print("\nRb:")
    for row in Rb:
        print(" ".join(map(str, row)))
    return Rb

def psi(vector):
    return sorted(vector, reverse=True)
#ВІДНОШЕННЯ ПОДИНОВСЬКОГО
def pod(matrix):
    print("\nВідношення Подиновського")
    n = len(matrix)
    psi_vectors = [psi(row) for row in matrix]
    print_matrix_pod(psi_vectors)
    Rp = pareto(psi_vectors)
    print("\nRp:")
    for row in Rp:
        print(" ".join(map(str, row)))
    return Rp


matrix = [
    [9, 4, 2, 2, 9, 10, 9, 3, 10, 3, 9, 10],
    [9, 4, 7, 3, 9, 10, 9, 3, 10, 4, 9, 10],
    [8, 4, 1, 2, 7, 7, 5, 3, 6, 3, 9, 10],
    [8, 4, 1, 1, 6, 4, 2, 3, 6, 3, 9, 10],
    [6, 4, 1, 1, 3, 4, 2, 3, 6, 3, 4, 1],
    [6, 1, 1, 1, 3, 4, 2, 3, 4, 2, 4, 1],
    [6, 1, 1, 1, 3, 3, 2, 3, 4, 2, 4, 1],
    [10, 9, 3, 7, 7, 8, 10, 3, 9, 5, 4, 3],
    [6, 4, 2, 2, 1, 4, 1, 3, 6, 3, 4, 3],
    [4, 4, 1, 2, 1, 4, 1, 3, 1, 3, 2, 3],
    [4, 1, 1, 1, 1, 4, 1, 3, 1, 2, 2, 1],
    [7, 6, 5, 8, 1, 4, 2, 9, 3, 2, 6, 7],
    [8, 6, 6, 8, 9, 4, 2, 9, 6, 3, 9, 10],
    [8, 6, 6, 8, 9, 3, 2, 9, 6, 3, 6, 10],
    [2, 4, 2, 1, 1, 3, 2, 4, 2, 2, 4, 7],
    [2, 4, 2, 1, 1, 3, 2, 2, 2, 2, 4, 4],
    [1, 4, 2, 1, 1, 2, 2, 2, 2, 2, 2, 4],
    [7, 8, 10, 6, 9, 5, 2, 7, 4, 10, 9, 8],
    [7, 2, 3, 3, 7, 5, 2, 2, 2, 5, 4, 3],
    [7, 7, 10, 3, 7, 5, 3, 7, 4, 8, 7, 3],
]


# Оптимізація 
# Функція для перевірки, чи є відношення асиметричним
def is_asymmetric(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1 and matrix[j][i] == 1:
                return False
    return True

# Функція для пошуку домінуючих альтернатив
def find_dominating_alternatives(matrix):
    print("\nОптимальні альтернативи за принципом домінування:")
    is_asymetric = is_asymmetric(matrix)
    
    if is_asymetric:
        print("Відношення асиметричне.")
        X_p = []
        for i in range(len(matrix)):
            for j, el in enumerate(matrix[i]):
                if i == j and el != 0:
                    break
                if i != j and el != 1:
                    break
            else:
                X_p.append(i + 1)
        print(f"X*P: {{{', '.join(map(str, X_p)) if X_p else '∅'}}}")
    else:
        print(f"Відношення не асиметричне")
        X_r = []
        for i in range(len(matrix)):
            if all(matrix[i]):
                X_r.append(i + 1)

        X__r = []
        for i in range(len(matrix)):
            if all(matrix[i]):
                for j in range(len(matrix)):
                    if i != j and matrix[j][i] == 1:
                        break
                else:
                    X__r.append(i + 1)
        print(f"X*R: {{{', '.join(map(str, X_r)) if X_r else '∅'}}}")
        print(f"X**R: {{{', '.join(map(str, X__r)) if X__r else '∅'}}}")

# Функція для пошуку блокованих альтернатив
def find_blocked_alternatives(matrix):
    print("\nОптимальні альтернативи за принципом блокування:")
    is_asymetric = is_asymmetric(matrix)
    
    if is_asymetric:
        print("Відношення асиметричне.")
        x_p = []
        for column_j in range(len(matrix[0])):
            for row_i in range(len(matrix)):
                if matrix[row_i][column_j] == 1:
                    break
            else:
                x_p.append(column_j + 1)
        print(f"X⁰P: {{{', '.join(map(str, x_p)) if x_p else '∅'}}}")
    else:
        print("Відношення не асиметричне")
        x_r = []
        for column_j in range(len(matrix[0])):
            for row_i in range(len(matrix)):
                if matrix[row_i][column_j] == 1 and matrix[column_j][row_i] == 0:
                    break
            else:
                x_r.append(column_j + 1)

        x__r = []
        for column_j in range(len(matrix[0])):
            for row_i in range(len(matrix)):
                if row_i != column_j and matrix[row_i][column_j] == 1:
                    break
            else:
                x__r.append(column_j + 1)
        print(f"X⁰R: {{{', '.join(map(str, x_r)) if x_r else '∅'}}}")
        print(f"X⁰⁰R: {{{', '.join(map(str, x__r)) if x__r else '∅'}}}")


def result():
    print_matrix(matrix)
    R0 = pareto(matrix)
    find_dominating_alternatives(R0)
    find_blocked_alternatives(R0)
    Rm = mag(matrix)
    find_dominating_alternatives(Rm)
    find_blocked_alternatives(Rm)
    Rl = lek(matrix)
    find_dominating_alternatives(Rl)
    find_blocked_alternatives(Rl)
    Rb = ber(matrix)
    find_dominating_alternatives(Rb)
    find_blocked_alternatives(Rb)
    Rp = pod(matrix)
    find_dominating_alternatives(Rp)
    find_blocked_alternatives(Rp)

result()



