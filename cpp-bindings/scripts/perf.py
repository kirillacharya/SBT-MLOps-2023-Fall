from typing import List, Callable
import numpy as np
import time
import linalg
import torch



def py_matrix_multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    if len(mat1[0]) != len(mat2):
        raise ValueError("Uncompatible matrix shapes")
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            sum_val = 0
            for k in range(len(mat2)):
                sum_val += mat1[i][k] * mat2[k][j]
            row_result.append(sum_val)
        result.append(row_result)
    return result


def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


def compare(matrix_size: int) -> None:
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    list_a = matrix_a.tolist()
    list_b = matrix_b.tolist()

    print(
        "Mat mul (Pure Python), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(py_matrix_multiply, list_a, list_b)
        )
    )
    print(
        "Mat mul (Pure C++), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulPure, list_a, list_b)
        )
    )
    print(
        "Mat mul (C++ BLAS), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulBlas, list_a, list_b)
        )
    )
    print(
        "Mat mul (Python numpy), size={0}x{0}: {1} seconds\n".format(
            matrix_size, test_timings(np.dot, np.array(list_a), np.array(list_b))
        )
    )

def cosine_similarity_compare(N : int, D : int) -> None:
    vec_set_a = torch.randn(N, D)
    vec_set_b = torch.randn(N, D)

    lst_a = vec_set_a.tolist()
    lst_b = vec_set_b.tolist()

    assert np.allclose(
        linalg.LinearAlgebra.cosineSimilarityBlas(np.array(lst_a), np.array(lst_b), 1e-6),
        torch.nn.functional.cosine_similarity(vec_set_a,vec_set_b), 1e-4)
    
    print(
        "Cosine similarity (C++ BLAS), size={0}x{1}: {2} seconds\n".format(
            N, D, test_timings(linalg.LinearAlgebra.cosineSimilarityBlas,
                               np.array(lst_a), np.array(lst_b), 1e-6)
        )
    )

    print(
        "Cosine similarity (torch), size={0}x{1}: {2} seconds\n".format(
            N, D, test_timings(torch.nn.functional.cosine_similarity,
                               vec_set_a, vec_set_b)
        )
    )


if __name__ == "__main__":
    for size in [100, 300, 500, 700, 1500, 3000]:
        cosine_similarity_compare(size, size)
