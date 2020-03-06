from numpy import *

lbp_matrix = array([1, 2, 4, 128, 0, 8, 64, 32, 16]).reshape(3, 3)


def local_binary_pattern(pattern) -> list:
    pattern_vector = zeros(2 ** 8, dtype=int)
    for row in range(1, pattern.shape[0] - 1):
        for column in range(1, pattern.shape[1] - 1):
            pattern_vector[_lbp_operation(pattern, row, column)] += 1
    return pattern_vector


def _lbp_operation(pattern, center_row, center_col) -> int:
    to_sum_matrix = [lbp_matrix[row_inc][col_inc]
                     for row_inc in range(0, 3)
                     for col_inc in range(0, 3)
                     if pattern[(center_row - 1) + row_inc][(center_col - 1) + col_inc]
                     >= pattern[center_row][center_col]]
    return sum(to_sum_matrix)
