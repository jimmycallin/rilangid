from pydsm import IndexMatrix
from pydsm.model import DSM
import numpy as np
import zlib

class RILangVectorAddition(DSM):

    def __init__(self,
                 corpus,
                 window_size,
                 config=None,
                 lower_threshold=None,
                 higher_threshold=None,
                 dimensionality=2000,
                 num_indices=8,
                 vocabulary=None,
                 matrix=None,
                 ordered=False,
                 directed=False,
                 language=None,
                 **kwargs):
        """
        Builds a Random Indexing DSM from text-iterator. 
        Parameters:
        window_size: 2-tuple of size of the context
        matrix: Instantiate DSM with already created matrix.
        vocabulary: When building, the DSM also creates a frequency dictionary. 
                    If you include a matrix, you also might want to include a frequency dictionary
        lower_threshold: Minimum frequency of word for it to be included.
        higher_threshold: Maximum frequency of word for it to be included.
        ordered: Differentates between context words in different positions. 
        directed: Differentiates between left and right context words.
        dimensionality: Number of columns in matrix.
        num_indices: Number of positive indices, as well as number of negative indices.
        """
        super(type(self), self).__init__(matrix,
                                         corpus,
                                         window_size,
                                         vocabulary,
                                         config,
                                         language=language,
                                         lower_threshold=lower_threshold,
                                         higher_threshold=higher_threshold,
                                         dimensionality=dimensionality,
                                         num_indices=num_indices,
                                         ordered=ordered,
                                         directed=directed)

    def _build(self, text):
        """
        Builds the co-occurrence dict from text.
        """
        word_to_col = {}
        indices = np.arange(self.config['dimensionality'])

        def get_index_vector(context):
            if context not in word_to_col:
                # Create index vector if not exist
                seed = zlib.adler32(context.encode())
                np.random.seed(seed)
                rand_indices = np.random.permutation(indices)
                pos_indices = rand_indices[rand_indices.size // 2:]
                neg_indices = rand_indices[:rand_indices.size // 2]
                index_vector = np.zeros_like(indices)
                np.add.at(index_vector, pos_indices, 1)
                np.add.at(index_vector, neg_indices, -1)
                word_to_col[context] = index_vector
            return word_to_col[context]

        remove_next = None
        language_vector = np.zeros((1, self.config['dimensionality']))
        for _, block in text:
            if not block:
                continue
            if not remove_next:
                block_vector = np.zeros_like(language_vector)[0]
                for context in block:
                    block_vector += get_index_vector(context)
            else:
                block_vector += get_index_vector(
                    block[-1]) - get_index_vector(remove_next)

            remove_next = block[0]
            language_vector += block_vector

        row2word = [self.config['language']]
        values = language_vector
        col2word = indices.tolist()

        return values, row2word, col2word

class RILangVectorPermutation(DSM):

    def __init__(self,
                 corpus,
                 window_size,
                 config=None,
                 lower_threshold=None,
                 higher_threshold=None,
                 dimensionality=2000,
                 num_indices=8,
                 vocabulary=None,
                 matrix=None,
                 ordered=False,
                 directed=False,
                 language=None,
                 is_ngrams=False,
                 word_vectors=None,
                 **kwargs):
        """
        This is a reproduction of [...] Kanerva (2014) Language recognition using Random Indexing.
        It uses multiplications of character index vectors to form character ngram vectors, aka block vectors.
        These are later added together to form text vectors, which are compared to language vectors. 
        The language vector that is closest to the text vector should be the language of the text.
        Parameters:
        window_size: 2-tuple of size of the context
        matrix: Instantiate DSM with already created matrix.
        vocabulary: When building, the DSM also creates a frequency dictionary. 
                    If you include a matrix, you also might want to include a frequency dictionary
        lower_threshold: Minimum frequency of word for it to be included.
        higher_threshold: Maximum frequency of word for it to be included.
        ordered: Differentates between context words in different positions. 
        directed: Differentiates between left and right context words.
        dimensionality: Number of columns in matrix.
        num_indices: Number of positive indices, as well as number of negative indices.
        """
        if word_vectors:
            self.word_vectors = word_vectors
        else:
            self.word_vectors = {}

        super(type(self), self).__init__(matrix,
                                         None,
                                         window_size,
                                         vocabulary,
                                         config,
                                         language=language,
                                         lower_threshold=lower_threshold,
                                         higher_threshold=higher_threshold,
                                         dimensionality=dimensionality,
                                         num_indices=num_indices,
                                         ordered=ordered,
                                         directed=directed,
                                         is_ngrams=is_ngrams)
    
        if corpus:
            self.matrix = IndexMatrix(*self._build(corpus))

    def _build(self, block_stream):
        """
        Builds the co-occurrence dict from text.
        """
        word_to_col = {}
        indices = np.arange(self.config['num_indices'])

        def get_index_vector(context):
            if context not in self.word_vectors:
                print("New word: {}".format(context))
                # Create index vector if not exist
                seed = zlib.adler32(context.encode()) # Deterministic seed
                np.random.seed(seed)
                rand_indices = np.random.permutation(self.config['num_indices'])
                pos_indices = rand_indices[rand_indices.size // 2:]
                neg_indices = rand_indices[:rand_indices.size // 2]
                index_vector = np.zeros((1,self.config['dimensionality']))
                index_vector[0,pos_indices] = 1
                index_vector[0,neg_indices] = -1
                self.word_vectors[context] = index_vector
            return self.word_vectors[context]

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            for k, char in enumerate(block):
                char = "{}_{}".format(char, k)
                block_vector = np.multiply(block_vector,get_index_vector(char))

            text_vector += block_vector

        row2word = [self.config['language']]
        col2word = indices.tolist()

        return text_vector, row2word, col2word


class RILangVectorConvolution(DSM):

    def __init__(self,
                 corpus,
                 window_size,
                 config=None,
                 lower_threshold=None,
                 higher_threshold=None,
                 dimensionality=2000,
                 num_indices=8,
                 vocabulary=None,
                 matrix=None,
                 ordered=False,
                 directed=False,
                 language=None,
                 word_vectors=None,
                 is_ngrams=False,
                 **kwargs):
        """
        This is a reproduction of [...] Kanerva (2014) Language recognition using Random Indexing.
        It uses multiplications of character index vectors to form character ngram vectors, aka block vectors.
        These are later added together to form text vectors, which are compared to language vectors. 
        The language vector that is closest to the text vector should be the language of the text.
        Parameters:
        window_size: 2-tuple of size of the context
        matrix: Instantiate DSM with already created matrix.
        vocabulary: When building, the DSM also creates a frequency dictionary. 
                    If you include a matrix, you also might want to include a frequency dictionary
        lower_threshold: Minimum frequency of word for it to be included.
        higher_threshold: Maximum frequency of word for it to be included.
        ordered: Differentates between context words in different positions. 
        directed: Differentiates between left and right context words.
        dimensionality: Number of columns in matrix.
        num_indices: Number of positive indices, as well as number of negative indices.
        """
        if word_vectors:
            self.word_vectors = word_vectors
        else:
            self.word_vectors = {}

        super(type(self), self).__init__(matrix,
                                         None,
                                         window_size,
                                         vocabulary,
                                         config,
                                         language=language,
                                         lower_threshold=lower_threshold,
                                         higher_threshold=higher_threshold,
                                         dimensionality=dimensionality,
                                         num_indices=num_indices,
                                         ordered=ordered,
                                         directed=directed,
                                         is_ngrams=is_ngrams)
    
        if corpus:
            self.matrix = IndexMatrix(*self._build(corpus))


    def _build(self, block_stream):
        """
        Builds the co-occurrence dict from text.
        """

        def get_index_vector(context):
            context = context[0]
            if context not in self.word_vectors:
                print("New word: {}".format(context))
                # Create index vector if not exist
                seed = zlib.adler32(context.encode()) # Deterministic seed
                np.random.seed(seed)
                rand_indices = np.random.permutation(self.config['num_indices'])
                pos_indices = rand_indices[rand_indices.size // 2:]
                neg_indices = rand_indices[:rand_indices.size // 2]
                index_vector = np.zeros((1,self.config['dimensionality']))
                index_vector[0,pos_indices] = 1
                index_vector[0,neg_indices] = -1
                self.word_vectors[context] = index_vector
            return self.word_vectors[context]

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            #print("Block {}: {}".format('init', block_vector))
            for k, char in enumerate(block):
                block_vector = np.multiply(block_vector,get_index_vector(char))
                block_vector = np.roll(block_vector, 1)
                #print("Block {}: {}".format(k, block_vector))

            text_vector += block_vector

        row2word = [self.config['language']]
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word