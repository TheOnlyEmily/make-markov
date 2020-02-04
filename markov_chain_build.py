import numpy as np


class MarkovChain:

    def __init__(self, generator_alphabet, edge_list=None):
        assert all((l is not None) for l in generator_alphabet)
        assert len(generator_alphabet) == len(set(generator_alphabet))
        self._gen_alphabet = np.array(generator_alphabet)
        alphabet_size = len(self._gen_alphabet)
        self._prob_mat = np.zeros(2 * [alphabet_size])
        self._mat_normalizer = 1
        if edge_list:
            self.update_from_edge_list(edge_list)

    def update_from_edge_list(self, edge_list):
        assert all((len(edge) == 2) for edge in edge_list)
        for e1, e2 in edge_list:
            self.update_from_edge(e1, e2)

    def update_from_edge(self, e1, e2):
        assert e1 in self._gen_alphabet
        assert e2 in self._gen_alphabet
        e1_index = self._get_prob_mat_index_for_(e1)
        e2_index = self._get_prob_mat_index_for_(e2)
        self._increment_prob_mat_cell(e1_index, e2_index)

    def _get_prob_mat_index_for_(self, i):
        return np.where(i == self._gen_alphabet)[0]

    def _increment_prob_mat_cell(self, i1, i2):
        value_at_index = self._prob_mat[i1, i2]
        value_at_index += 1
        self._mat_normalizer = max(self._mat_normalizer, value_at_index)
        self._prob_mat[i1, i2] = value_at_index

    def generate_sequence(self, seq_length):
        assert type(seq_length) is int
        prob_vector = self._get_equally_likely_prob_vect()
        return_sequence = []
        for _ in range(seq_length):
            next = np.random.choice(self._gen_alphabet, 1, p=prob_vector)
            return_sequence.extend(next)
            prob_vector = self._generate_prob_vect_from_prob_vect(prob_vector)
        return return_sequence


    def generate_sequence_starting_with_(self, seq_start, seq_length):
        assert seq_start in self._gen_alphabet
        assert type(seq_length) is int
        prob_vector = self._get_single_result_prob_vector(seq_start)
        return_sequence = []
        for _ in range(seq_length):
            next = np.random.choice(self._gen_alphabet, 1, p=prob_vector)
            return_sequence.extend(next)
            prob_vector = self._generate_prob_vect_from_prob_vect(prob_vector)
        return return_sequence

    def _generate_prob_vect_from_prob_vect(self, prob_vector):
        assert type(prob_vector) is np.ndarray
        assert prob_vector.size == self._prob_mat.size[0]
        assert prob_vector.size == self._prob_mat.size[1]
        assert np.sum(prob_vector) == 1
        assert np.all((p <= 1) for p in prob_vector)
        return np.dot(prob_vector, self._prob_mat / self._mat_normalizer)

    def _get_equally_likely_prob_vect(self):
        gen_alpha_size = self._gen_alphabet.size
        return np.ones(gen_alpha_size) / gen_alpha_size

    def _get_single_result_prob_vector(self, v):
        assert v in self._gen_alphabet
        return (self._gen_alphabet == v).astype(int)