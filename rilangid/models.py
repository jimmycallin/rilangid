from pydsm import IndexMatrix
from pydsm.model import DSM
import numpy as np
import hashlib

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
                 is_ngrams=False,
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

    def _build(self, text):
        """
        Builds the co-occurrence dict from text.
        """
        word_to_col = {}

        def get_index_vector(context):
            context = context[0]
            if context not in self.word_vectors:
                # Create index vector if not exist
                hsh = hashlib.sha1()
                hsh.update(context.encode())
                seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
                np.random.seed(seed)
                rand_indices = np.random.permutation(self.config['num_indices'])
                pos_indices = rand_indices[rand_indices.size // 2:]
                neg_indices = rand_indices[:rand_indices.size // 2]
                index_vector = np.zeros((1,self.config['dimensionality']))
                index_vector[0,pos_indices] = 1
                index_vector[0,neg_indices] = -1
                self.word_vectors[context] = index_vector
            return self.word_vectors[context]

        language_vector = np.zeros((1, self.config['dimensionality']))
        for block in text:

            block_vector = np.zeros_like(language_vector)
            for context in block:
                block_vector += get_index_vector(context)

            language_vector += block_vector

        row2word = [self.config['language']]
        col2word = list(range(self.config['dimensionality']))

        return language_vector, row2word, col2word

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
                # Create index vector if not exist
                hsh = hashlib.sha1()
                hsh.update(context.encode())
                seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
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
                # Create index vector if not exist
                hsh = hashlib.sha1()
                hsh.update(context.encode())
                seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
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


class RILangVectorConvolutionNgrams(DSM):

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
            hsh = hashlib.sha1()
            hsh.update(context.encode())
            seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
            np.random.seed(seed)
            rand_indices = np.random.permutation(self.config['num_indices'])
            pos_indices = rand_indices[rand_indices.size // 2:]
            neg_indices = rand_indices[:rand_indices.size // 2]
            index_vector = np.zeros((1,self.config['dimensionality']))
            index_vector[0,pos_indices] = 1
            index_vector[0,neg_indices] = -1
            return index_vector

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            #print("Block {}: {}".format('init', block_vector))
            block_vector = np.multiply(block_vector,get_index_vector("".join(block)))
            block_vector = np.roll(block_vector, 1)

            text_vector += block_vector

        row2word = [self.config['language']]
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class ShortestPathLI(DSM):

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
                 **kwargs):
        """

        """

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
            hsh = hashlib.sha1()
            hsh.update(context.encode())
            seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
            np.random.seed(seed)
            rand_indices = np.random.permutation(self.config['num_indices'])
            pos_indices = rand_indices[rand_indices.size // 2:]
            neg_indices = rand_indices[:rand_indices.size // 2]
            index_vector = np.zeros((1,self.config['dimensionality']))
            index_vector[0,pos_indices] = 1
            index_vector[0,neg_indices] = -1
            return index_vector

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            #print("Block {}: {}".format('init', block_vector))
            block_vector = np.multiply(block_vector,get_index_vector("".join(block)))
            block_vector = np.roll(block_vector, 1)

            text_vector += block_vector

        row2word = [self.config['language']]
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word

class RICharacterNgramVectors(DSM):

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
        block_vectors = {}
        def get_index_vector(context):
            if context not in self.word_vectors:
                # Create index vector if not exist
                hsh = hashlib.sha1()
                hsh.update(context.encode())
                seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
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
            if "".join(block) not in block_vectors:
                block_vector = np.ones_like(text_vector)
                for char in block:
                    block_vector = np.multiply(block_vector,get_index_vector(char))
                    block_vector = np.roll(block_vector, 1)

                block_vectors["".join(block)] = block_vector


        row2word = list(block_vectors.keys())
        col2word = indices.tolist()
        vectors = np.vstack(block_vectors.values())

        return vectors, row2word, col2word


class RIIndexVectors(DSM):

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
        block_vectors = {}
        def get_index_vector(context):
            if context not in self.word_vectors:
                # Create index vector if not exist
                hsh = hashlib.sha1()
                hsh.update(context.encode())
                seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
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
            word = "".join(block)
            if word not in block_vectors:
                block_vectors[word] = get_index_vector(word)


        row2word = list(block_vectors.keys())
        col2word = indices.tolist()
        vectors = np.vstack(block_vectors.values())

        return vectors, row2word, col2word