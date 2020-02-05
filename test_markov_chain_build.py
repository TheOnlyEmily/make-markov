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


class TestSecondaryHelperMethods:

    def test_update_from_edge(self):
        expected_matrix = np.array([[0, 1], [0, 0]])

        mc = MarkovChain(('a', 'b'))
        mc.update_from_edge('a', 'b')

        assert np.all(mc._prob_mat == expected_matrix)

    def test_generate_probability_vector_from_probability_vector(self):
        mc1 = MarkovChain(('a', 'b'), [('a', 'a')])
        mc2 = MarkovChain(('a', 'b'), [('b', 'b')])

        a_vector = np.array([1, 0])
        b_vector = np.array([0, 1])

        assert np.all(mc1._prob_mat == np.array([[1, 0], [0, 0]]))
        assert np.all(mc2._prob_mat == np.array([[0, 0], [0, 1]]))

        assert np.all(mc1._generate_prob_vect_from_prob_vect(a_vector) == np.array([1, 0]))
        assert np.all(mc1._generate_prob_vect_from_prob_vect(b_vector) == np.array([0, 0]))
        assert np.all(mc2._generate_prob_vect_from_prob_vect(a_vector) == np.array([0, 0]))
        assert np.all(mc2._generate_prob_vect_from_prob_vect(b_vector) == np.array([0, 1]))


class TestPrimaryMethods:


        class TestInit:

            def test_without_edge_list_argument(self):
                mc = MarkovChain(('a', 'b'))

                assert np.all(mc._gen_alphabet == np.array(['a', 'b']))
                assert np.all(mc._prob_mat == np.array([[0, 0], [0, 0]]))
                assert np.all(mc._mat_normalizer == np.array([[0], [0]]))

            def test_with_edge_list_argument(self):
                mc = MarkovChain(('a', 'b'), [('a', 'a')])

                assert np.all(mc._gen_alphabet == np.array(['a', 'b']))
                assert np.all(mc._prob_mat == np.array([[1, 0], [0, 0]]))
                assert np.all(mc._mat_normalizer == np.array([[1], [0]]))


        class TestSequenceGeneartion:

            def test_without_specified_starting_character(self):
                mc = MarkovChain(('a', 'b'), [('a', 'a'), ('a', 'b'), ('b', 'a')])

                result = mc.generate_sequence(5)

                assert len(result) == 5
                assert all((e == 'a') or (e == 'b') for e in result)

            def test_with_specified_starting_character(self):
                mc = MarkovChain(('a', 'b'), [('a', 'a'), ('a', 'b'), ('b', 'a')])

                result = mc.generate_sequence('a', 5)

                assert result[0] == 'a'
                assert len(result) == 5
                assert all((e == 'a') or (e == 'b') for e in result)
