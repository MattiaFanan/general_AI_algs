from numpy import *

_lbp_operator = array([1, 2, 4, 128, 0, 8, 64, 32, 16]).reshape(3, 3)
_amino_dict = {amino: pos for pos, amino in enumerate("ARNDCGEQHILKMFPSTWYV")}


def local_binary_pattern(pattern) -> ndarray:
    pattern_vector = zeros(2 ** 8, dtype=int)
    for row in range(1, pattern.shape[0] - 1):
        for column in range(1, pattern.shape[1] - 1):
            pattern_vector[_apply_lbp_operator(pattern, row, column)] += 1
    return pattern_vector


def _apply_lbp_operator(pattern, center_row, center_col) -> int:
    to_sum_matrix = [_lbp_operator[row_inc][col_inc]
                     for row_inc in range(0, 3)
                     for col_inc in range(0, 3)
                     if _s_function(
            pattern[(center_row - 1) + row_inc][(center_col - 1) + col_inc],
            pattern[center_row][center_col])
                     ]
    return sum(to_sum_matrix)


def _s_function(first_term, second_term):
    return first_term >= second_term


def amino_acid_composition(protein):
    feature_vector = zeros(len(protein))
    for base in protein:
        feature_vector[_amino_dict[base]] += 1

    return [amino_count/len(protein) for amino_count in feature_vector]
