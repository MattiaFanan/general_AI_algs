from numpy import *
from unittest import TestCase
import ai_algs


class TestAlg(TestCase):

    def test_lbs(self):
        expected1 = zeros(2 ** 8, dtype=int)
        expected1[37] = 1

        param_matrix1 = array([
            [98, 88, 95],
            [25, 90, 75],
            [74, 92, 35]
        ])

        expected2 = zeros(2 ** 8, dtype=int)
        expected2[180] = 1
        expected2[238] = 1
        expected2[22] = 1
        expected2[112] = 1
        expected2[245] = 1
        expected2[255] = 1

        param_matrix2 = array([
            [1, 2, 5, 7, 8],
            [5, 4, 3, 7, 5],
            [2, 7, 4, 2, 8],
            [9, 14, 7, 9, 2]
        ])

        assert array_equiv(expected1, ai_algs.local_binary_pattern(param_matrix1)), "not equal lbp1"
        assert array_equiv(expected2, ai_algs.local_binary_pattern(param_matrix2)), "not equal lbp2"

    def test_amino_acid_composition(self):
        amino = "LNQAVSVAQARENFSRVEQA"
        expected = [4 / 20, 2 / 20, 2 / 20, 0, 0, 0, 2 / 20, 3 / 20, 0, 0,
                    1 / 20, 0, 0, 1 / 20, 0, 2 / 20, 0, 0, 0, 3 / 20]
        result = ai_algs.amino_acid_composition(amino)
        assert array_equiv(expected, result), "not equals amino"
