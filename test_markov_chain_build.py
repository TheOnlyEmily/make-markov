import pytest
import numpy as np

from markov_chain_build import MarkovChain


class TestBaseHelperMethods:


    class TestUpdateFromEdgeBase:

        def test_get_mat_prob_index_for(self):
            mc = MarkovChain(('a', 'b'))
            assert mc._get_prob_mat_index_for_('a') == 0
            assert mc._get_prob_mat_index_for_('b') == 1

        def test_increment_prob_mat_cell(self):
            A_INDEX = 0
            B_INDEX = 1

            initial_prob_matrix = np.array([[0, 0], [0, 0]])
            final_prob_matrix = np.array([[0, 1], [0, 0]])

            mc = MarkovChain(('a', 'b'))

            assert np.all(mc._prob_mat == initial_prob_matrix)

            mc._increment_prob_mat_cell(A_INDEX, B_INDEX)

            assert np.all(mc._prob_mat == final_prob_matrix)


    class TestGetProbabilityVector:

        def test_get_equally_likely_prob_vect(self):
            expected_vector = np.array([0.5, 0.5])
            mc = MarkovChain(('a', 'b'))
            assert np.all(mc._get_equally_likely_prob_vect() == expected_vector)

        def test_get_single_result_prob_vector(self):
            expected_vector = np.array([1, 0])
            mc = MarkovChain(('a', 'b'))
            assert np.all(mc._get_single_result_prob_vector('a') == expected_vector)
