import numpy as np



class MarkovChain:
    """
    Builds a simple Markov Chain, which can also be initialized from a list
    of symbols and the symbol that follows them in the data to be replicated
    by the MarkovChain.

    Arguments:
        generator_alphabet (iterable): all applicable symbols in sequences
        created by this class.

        edge_list (iterable | None): 'Training data' used to initialize the
        MarkovChain object.

    Raises:
        AssertionError: if generator_alphabet is empty or has no length method.
    """

    def __init__(self, generator_alphabet, edge_list=None):
        assert len(generator_alphabet) > 0
        self._gen_alphabet = np.array(generator_alphabet)
        alphabet_size = len(self._gen_alphabet)
        self._prob_mat = CoreProbabilityMatrix(alphabet_size)
        if edge_list:
            self.update_from_edge_list(edge_list)

    def update_from_edge_list(self, edge_list):
        """
        Takes a list of edges and updates the probabilities of one symbol
        appearing before the other in sequence.

        Arguments:
            edge_list (iterable): a list of edges or transitions found in the
            the data to be replicated by the MarkovChain.

        Raises:
            AssertionError: edge_list must consist of pairs of symbols
        """
        assert all((len(edge) == 2) for edge in edge_list)
        for e1, e2 in edge_list:
            self.update_from_edge(e1, e2)


    def update_from_edge(self, e1, e2):
        """
        Takes a starting node (e1) and an ending node (e2) and updates the
        probability of e1 following e2.

        Arguments:
            e1: The starting node in the Markov Chain.
            e2: The ending node in the Markov Chain.

        Raises:
            AssertionError: if e1 or e2 can't be found in the alphabet of
            symbols that are meant to be in replicated sequences.
        """
        assert e1 in self._gen_alphabet
        assert e2 in self._gen_alphabet
        e1_index = self._get_prob_mat_index_for_(e1)
        e2_index = self._get_prob_mat_index_for_(e2)
        self._prob_mat._increment_prob_mat_cell(e1_index, e2_index)

    def _get_prob_mat_index_for_(self, i):
        """
        An internal helper method for locating indexes in the internal
        matrix that stores the probabilities of various symbols appearing
        after or before one another.

        Arguments:
            i: A symbole to be found in replicated sequences.

        Raises:
            AssertionError: If i can not be found in the alphabet used to
            generate sequences from the source data.

        Returns:
            (int): The index of i in the internal matrix.
        """
        assert i in self._gen_alphabet
        return np.where(i == self._gen_alphabet)[0]

    def generate_sequence(self, seq_length):
        """
        Creates a semi-random list of symbols based on the edge list supplied
        previousely.

        Arguments:
            seq_length (int): The length of the sequence to be generated.

        Raises:
            AssertionError: If sequence length is not an integer.
            AssertionError: If no edges have been added.

        Returns:
            (list): A list of randomly selected symbols.
        """
        assert type(seq_length) is int
        prob_vector = self._get_equally_likely_prob_vect()
        return self._generate_sequence_from_prob_vector(prob_vector, seq_length)

    def generate_sequence_starting_with_(self, seq_start, seq_length):
        """
        Creates a semi-random sequence starting with a symbol in the alphabet
        used by the MarkovChain.

        Arguments:
            seq_start (*): The starting symbol in the semi-random sequence.

            seq_length (int): The lengthe of the semi-random sequence.

        Raises:
            AssertionError: If seq_length is not an integer.
            AssertionError: If seq_start is not in the Markov Chain alphabet.
            AssertionError: If no edges have been added.

        Returns:
            (list): A semi-random sequence starting with seq_start.
        """
        assert np.any(self._prob_mat._core_mat > 0)
        assert type(seq_length) is int
        assert seq_start in self._gen_alphabet
        prob_vector = self._get_single_result_prob_vector(seq_start)
        return self._generate_sequence_from_prob_vector(prob_vector, seq_length)


    def _generate_sequence_from_prob_vector(self, prob_vector, seq_length):
        """
        A helper method that is used by the two public methods for generating
        semi-random sequence. It takes starting prob_vector and the required
        seq_length, and delivers a semi-random sequence.

        Arguments:
            prob_vector (np.ndarray): An array of probabilities, indicating the
            probability of a symbol in the alphabet being selected to be in the
            semi-randomly generated sequence.

            seq_length (int): The length of the semi-randomly generated
            sequence.

        Raises:
            AssertionError: If the core probability martrix is all zeros.
            AssertionError: An element of prob_vector is less than zero or
            greater than 1.
            AssertionError: If seq_length is not an integer.
            AssertionError: The sum of prob_vector is not 1.
        """
        assert np.any(self._prob_mat._core_mat > 0)
        assert all((p >= 0) and (p <= 1) for p in prob_vector)
        assert type(seq_length) is int
        assert sum(prob_vector) == 1
        return_sequence = []
        for _ in range(seq_length):
            next = np.random.choice(self._gen_alphabet, 1, p=prob_vector)
            return_sequence.extend(next)
            prob_vector = self._prob_mat._generate_prob_vect_from_prob_vect(prob_vector)
        return return_sequence

    def _get_equally_likely_prob_vect(self):
        """
        Constructs a vector the same size of the alphabet. Where each symbol
        in the alphabet is equally likely given this vector.

        Returns:
            (np.ndarray): A propbability vector where each outcome represented
            in the vector is equally likely.
        """
        gen_alpha_size = self._gen_alphabet.size
        return np.ones(gen_alpha_size) / gen_alpha_size

    def _get_single_result_prob_vector(self, v):
        assert v in self._gen_alphabet
        return (self._gen_alphabet == v).astype(int)


class CoreProbabilityMatrix:
    """
    A helper object. Keeps a record of the probability of one symbol in
    MarkovChain's alphabet appearing after another. Facilitates adding
    of new data to the MarkovChain. Calcuates probabilites of any one
    symbol appearing after another.

    Arguments:
        alphabet_size (int): The number of symbols that the MarkovChain
        recognises as valid.
    """

    def __init__(self, alphabet_size):
        self._core_mat = np.zeros(2 * [alphabet_size])
        self._mat_normalizer = np.ones(alphabet_size).reshape((-1, 1))

    def _increment_prob_mat_cell(self, i1, i2):
        """
        Logs occurences of one symbol appearing after another.

        Arguments:
            i1 (int): An index representing the starting symbol in the
            MarkovChain alphabet.

            i2 (int): An index representing the ending symbol in the
            MarkovChain alphabet.
        """
        self._core_mat[i1, i2] += 1
        self._mat_normalizer = self._core_mat.sum(axis=1).reshape((-1, 1))
        zero_locations = self._mat_normalizer == 0
        self._mat_normalizer[zero_locations] = 1

    def _generate_prob_vect_from_prob_vect(self, prob_vector):
        """
        Calculates the probabilites of symbols in the MarkovChain alphabet
        being included in a generated sequence.

        Arguments:
            prob_vector (np.ndarray): The previouse probabilites of each symbol
            in the MarkovChain alphabet appearing in a generated sequence.

        Returns:
            (np.ndarray): The new probabilites of each symbol in the MarkovChain
            alphabet appearing in a generated sequence.

        Raises:
            AssertionError: If prob_vector is not an np.ndarray.
            AssertionError: If the prob_vector is of incorrect size.
            AssertionError: If prob_vector does not sum to one.
            AssertionError: If a value in the prob_vector is less than zero
            or greater than one.
        """
        assert type(prob_vector) is np.ndarray
        assert prob_vector.size == self._core_mat.shape[0]
        assert prob_vector.size == self._core_mat.shape[1]
        assert np.sum(prob_vector) == 1
        assert np.all((p <= 1) and (p >= 0) for p in prob_vector)
        return np.dot(prob_vector, self._core_mat / self._mat_normalizer)
