import numpy as np
from numba import njit

def remove_duplicates(arr1, arr2):
    unique_vals, inverse_idx, counts = np.unique(arr1, return_inverse=True, return_counts=True)

    # Sum and count arr2 values for each unique arr1 value
    sum_arr2 = np.zeros_like(unique_vals, dtype=float)
    count_arr2 = np.zeros_like(unique_vals, dtype=float)

    for i, val in enumerate(arr2):
        sum_arr2[inverse_idx[i]] += val
        count_arr2[inverse_idx[i]] += 1

    mean_arr2 = sum_arr2 / count_arr2

    arr1_no_duplicates = unique_vals
    arr2_no_duplicates = mean_arr2

    return arr1_no_duplicates, arr2_no_duplicates

@njit
def solve_lu(lu, piv, b):
    n = len(b)
    x = np.zeros(n)

    # Forward substitution
    for i in range(n):
        x[i] = b[piv[i]]
        for j in range(i):
            x[i] -= lu[i, j] * x[j]

    # Backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= lu[i, j] * x[j]
        x[i] /= lu[i, i]

    return x
