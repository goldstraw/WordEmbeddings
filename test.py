import random
import math

def dot_product(matrix, vector):
    result = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]
    return result

def norm(vector):
    squared_sum = sum([x**2 for x in vector])
    return math.sqrt(squared_sum)

def normalize(vector):
    vector_norm = norm(vector)
    return [x / vector_norm for x in vector]

def power_iteration(A, num_iterations, num_eigenvalues):
    eigenvalues = []
    eigenvectors = []
    prev_eigenvalue = 9**9

    for _ in range(num_eigenvalues):
        b_k = [random.random() for _ in range(len(A[0]))]

        for _ in range(num_iterations):
            b_k1 = dot_product(A, b_k)
            b_k1_norm = norm(b_k1)
            b_k = normalize(b_k1)

        temp = dot_product(A, b_k)
        eigenvalue = prev_eigenvalue if prev_eigenvalue != 9**9 else 0
        for i in range(len(b_k)):
            eigenvalue += b_k[i] * temp[i]
        eigenvectors.append(b_k)
        eigenvalues.append(eigenvalue)

        # Deflate the matrix A
        # A -= [[eigenvalue * b_k[j] * b_k[i] for j in range(len(b_k))] for i in range(len(b_k))]
        for i in range(len(b_k)):
            A[i][i] -= eigenvalue
        if prev_eigenvalue == 9**9:
            prev_eigenvalue = eigenvalue

    return eigenvalues, eigenvectors
print(power_iteration([[28,120,54,27],[40,23,29,23],[52,23,10,4],[8,3,6,15]], 1000, 4))

