import pydsm
from pydsm import IndexMatrix, RandomIndexing
from pydsm.model import DSM
import scipy
import numpy as np
import hashlib


class RILangID(RandomIndexing):

    def __init__(self,
                 corpus=None,
                 config=None,
                 **kwargs):
        """
        This is a super class for all language identification models using random indexing.
        Its only purpose is to initiate the RandomIndexing.__init__ using the given configuration.
        """
        super().__init__(config=config)


class LetterBased(RILangID):

    """
    This is a superclass to all language identification models that are based on character ngram distributions.
    It mainly specifies a training function and a language identification function.
    """

    @classmethod
    def words_to_blocks(cls, corpus, block_size):
        """
        Convert a text string to a block vector.
        Example:

        list(LetterBased.words_to_blocks('an example', 3))
        Out[5]:
        [['a', 'n', ' '],
         ['n', ' ', 'e'],
         [' ', 'e', 'x'],
         ['e', 'x', 'a'],
         ['x', 'a', 'm'],
         ['a', 'm', 'p'],
         ['m', 'p', 'l'],
         ['p', 'l', 'e']]
        """
        block = []
        for sentence in corpus:
            for char in sentence:
                if len(block) < block_size:
                    block.append(char)
                    continue
                yield block
                block = block[1:] + [char]
        yield block

    def __init__(self, corpus=None, config=None, **kwargs):
        super().__init__(corpus=corpus, config=config)
        self.word_vectors = {}

    def get_index_vector(self, context):
        """
        Returns an index vector given a word.
        An index vector is a sparse [dimensionality]-dimensional vector with [num_indices] random +1's and -1's.
        Usually, these are set to a dimensionality of 2000 with 8 indices.
        It saves the index vector in a dictionary for faster retrival.
        """
        if context not in self.word_vectors:
            # Create index vector if not exist
            hsh = hashlib.md5()
            hsh.update(context.encode())
            # highest number allowed by seed
            seed = int(hsh.hexdigest(), 16) % 4294967295
            np.random.seed(seed)
            rand_indices = np.random.permutation(
                self.config['dimensionality'])[:self.config['num_indices']]
            pos_indices = rand_indices[rand_indices.size // 2:]
            neg_indices = rand_indices[:rand_indices.size // 2]
            index_vector = np.zeros((1, self.config['dimensionality']))
            index_vector[0, pos_indices] = 1
            index_vector[0, neg_indices] = -1
            self.word_vectors[context] = index_vector
        return self.word_vectors[context]

    def identify(self, sentence):
        """
        Given a sentence, identify the language.
        This is done by building a context vector for the sentence,
        looking for the closest neighbor of the language vectors.
        """
        text_model = self.build(sentence)
        return self.nearest_neighbors(text_model).row2word[0]

    def train(self, corpora):
        """
        Trains a language model.
        Parameters:
            corpora: A dict of language: file objects.
        """

        print("Creating new model {}.".format(type(self).__name__))

        for i, (language, corpus) in enumerate(corpora.items()):
            print(
                "Reading {} ... {} / {}".format(language, i + 1, len(corpora)))
            self.config['language'] = language
            lang_vector = self.build(corpus)
            self.matrix = self.matrix.merge(lang_vector)

        del self.config['language']

        print("Done training model.")

    def build(self, text):
        """
        Returns an index matrix built by one of the subclasses.
        Each subclass therefore has to have a _build function.
        """
        return IndexMatrix(*self._build(self.words_to_blocks(text, self.config['block_size'])))


# # # # # # Implemented models # # # # # #

class RILangVectorAddition(LetterBased):

    """
    A letter based model where all block vectors are simply composed by addition.
    This produces bad results:
        Precision: 0.7545469776746799
        Recall: 0.7430912902611015
        F-score: 0.7419435479419189
    """

    def _build(self, text):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in text:
            block_vector = np.zeros_like(text_vector)
            for k, char in enumerate(block):
                block_vector += self.get_index_vector(char)

            text_vector += block_vector

        row2word = [self.config['language']] if 'language' in self.config else ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorConvolution(LetterBased):

    """
    This is a reimplementation of the algorithm explained in [1].
    The main algorithm is basically elementwise multiplication of permuted character n-gram vectors of a sentence.
    The character ngram vectors (blocks) are added together to form a text vector,
    which is later compared to language vectors that are formed in the same manner, but on more data.

    Their best results were retrieved by using: Window size: 3+0
    Dimensionality: 10 000
    Num 1/-1: 10 000 (5k each)

    [1] Joshi et. al. "Language Recognition using Random Indexing" (2014)
        http://arxiv.org/abs/1412.7026
    """

    def _build(self, block_stream):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            for k, char in enumerate(block):
                block_vector = np.multiply(
                    block_vector, self.get_index_vector(char))
                block_vector = np.roll(block_vector, 1)

            text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorPermutation(LetterBased):

    """
    This model is almost the same as RILangVectorConvolution,
    but instead of performing a convolution (roll) of the vector to determine character position,
    simply generate new index vectors unique for each position in relation to the focus word.
    This technique is explained in [1].

    [1] Sahlgren et. al. "Permutations as a means to encode order in word space." (2008).
    """

    def _build(self, block_stream):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            for k, char in enumerate(block):
                char = "{}_{}".format(char, k)
                block_vector = np.multiply(block_vector, self.get_index_vector(char))

            text_vector += block_vector

        row2word = [self.config['language']] if 'language' in self.config else ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorNgrams(LetterBased):

    """
    The idea is that elementwise multiplication of dense vectors adds no new information to the sum of its parts,
    but would rather just create another orthogonal vectors in the vector space. If this is the case, you would be
    able to achieve comparable results by skipping the character composition step of the RI algorithm.

    Instead of building block vectors from character ngrams by the means of permuting or convoluting character vectors,
    simply create vectors from the ngram directly. This gives equivalent results as RILangVectorConvolution, and proves
    that character composition to block vectors don't really do anything.

    Results:
    Precision: 0.9757865109248872
    Recall: 0.9730798551553268
    F-score: 0.9743154885493124
    """

    def _build(self, text_model):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in text_model:
            block_vector = self.get_index_vector("".join(block))
            text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class MultipleNgrams(RILangID):

    """
    This model is a preliminary test to see if using varying number of character ngrams is a viable method.
    It didn't produce better results.
    """

    def _build(self, text_model):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in text_model:
            n = len(block) // 2
            parts = ["".join(block[:n]), "".join(block[n:]), "".join(block)]
            for part in parts:
                block_vector = self.get_index_vector(part)
                text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorNgramsGramSchmidt(RILangVectorNgrams):

    """
    Perform the build in RILangVectorNgrams, and thereafter run Gram-Schmidt on the matrix for ortohogonalization.

    By orthogonalizing the vectors in the language space, we believed it would wash off common features among the
    languages. Left is an orthonormal vector space. The text vector will be projected onto the vectors in the space.

    Unfortunately, this turned out to not work that well.
    Precision: 0.8455455033071209
    Recall: 0.6270249666476082
    F-score: 0.6269619636831721

    My best guess is that Gram-Schmidt moves around the vectors too much, which makes it hard for the text vector to
    keep track of the changes. If there was a way to perform something similar to the orthogonalization onto the
    text vector, perhaps it would get closer to the language in question.

    """

    @DSM.matrix.setter
    def matrix(self, mat):
        """
        When setting the matrix of the DSM, perform the ortohogonalization first.
        """
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            self._matrix = mat
        else:
            orthogonalized = scipy.sparse.coo_matrix(
                self.gram_schmidt(mat.to_ndarray()))
            self._matrix = mat._new_instance(orthogonalized)

    def nearest_neighbors(self, text_vec, sim_func=pydsm.similarity.cos):
        """
        Project vector on language vectors before performing nearest neighbors search.
        """
        lang_vecs = self.matrix
        scalar_projections = (
            lang_vecs.dot(text_vec.transpose()) / lang_vecs.norm(axis=1))
        projected = lang_vecs.multiply(scalar_projections).sum(axis=0)

        return DSM.nearest_neighbors(self, projected, sim_func=sim_func)

    @classmethod
    def gram_schmidt(cls, vecs, row_wise_storage=True):
        """
        Apply the Gram-Schmidt orthogonalization algorithm to a set
        of vectors. vecs is a two-dimensional array where the vectors
        are stored row-wise, or vecs may be a list of vectors, where
        each vector can be a list or a one-dimensional array.
        An array basis is returned, where basis[i,:] (row_wise_storage
        is True) or basis[:,i] (row_wise_storage is False) is the i-th
        orthonormal vector in the basis.
        """
        from numpy.linalg import inv
        vecs = np.asarray(vecs)  # transform to array if list of vectors
        m, n = vecs.shape
        basis = np.array(np.transpose(vecs))
        eye = np.identity(n).astype(float)

        basis[:, 0] /= np.sqrt(np.dot(basis[:, 0], basis[:, 0]))
        for i in range(1, m):
            v = basis[:, i] / np.sqrt(np.dot(basis[:, i], basis[:, i]))
            U = basis[:, :i]
            P = eye - \
                np.dot(
                    U, np.dot(inv(np.dot(np.transpose(U), U)), np.transpose(U)))
            basis[:, i] = np.dot(P, v)
            basis[:, i] /= np.sqrt(np.dot(basis[:, i], basis[:, i]))

        return np.transpose(basis) if row_wise_storage else basis


class ShortestPath(RILangID):

    """
    This is the first basic idea I had about lang id with RI. The essential idea is to build
    syntagmatic word spaces for each language, and compare the "traveled distance" of a sentence
    by measuring the cosine for word n/skip-grams. If a word is unknown for a wordspace, it's considered
    to be orthogonal to all other words. The wordspace/language that has the shortest distance is the winner.

    Because of languages lacking word boundaries, it might be necessary to look at most on character-ngrams.
    This won't be necessary for the given test data, but if we were to implement it later, such a model should
    be taken into consideration. It would essentially work by splitting each word into n-grams, while simultaneously
    treating them as a whole unit: [the, cat, sat, on, the, [blu, lue], mat]. [blu, lue] will have the same
    context window, but won't take into consideration each other.

    This beats the RILangVectorConvolution model.
    Precision: 0.9968937117674463
    Recall: 0.9917571945873833
    F-score: 0.9943000225552215

    Using config: Ordered permutations, window size of 100+100 (syntagmatic), proobably significant, increasing by one percentage point:
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def identify(self, sentence):
        words = sentence.split()
        best_lang = None
        best_score = 0

        for language, langmat in self.matrix.items():
            last_vec = None
            distance = 0

            for w in words:
                if w in langmat.word2row:
                    vec = langmat[w]
                else:
                    vec = None

                if last_vec and vec:
                    distance += abs(pydsm.similarity.cos(
                                    last_vec,
                                    vec,
                                    assure_consistency=self.config.get('assure_consistency',
                                                                       False))[0, 0])
                last_vec = vec

            if distance > best_score:
                best_score = distance
                best_lang = language

        if best_lang is None:
            print("NONE: {} with score of {} for sentence: \n {} \n {}".format(best_lang, best_score, sentence, words))
        return best_lang

    def train(self, corpora):
        self.matrix = {}
        for language, corpus in corpora.items():
            print("Reading {}...".format(language))
            self.matrix[language] = self.build(corpus)

    def build(self, text):
        return RandomIndexing(corpus=text, config=self.config).matrix


class Eigenvectors(RILangID):

    """
    The idea behind Eigenvectors is that we create a language space by creating random indexing matrices for
    each language. Each RI matrix is then collapsed into a vector by summing all columns. This vector will
    be an approximation to the first eigenvector, according to the power iteration matrix.
    http://en.wikipedia.org/wiki/Power_iteration

    By doing this for each language, we get a language space of one language vector per language being the
    first eigenvectors. When identifying a sentence, we create a sentence vector of the sentence for each language.
    A sentence vector that is very close the eigenvector for the given language should mean that the sentence is very
    similar to the langugage. As such, the sentence with the smallest distance from its language is the determined
    language.

    This doesn't really seem to work very well.

    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.langvectors = IndexMatrix({})
        self.unknown_vec = self.create_unknown()

    def create_unknown(self):
        """
        This is a vector that is orthogonal to all other vectors.
        This is used for words that are unknown to the model.
        """
        # Create index vector if not exist
        hsh = hashlib.md5()
        hsh.update("$UNKNOWN$".encode())
        # highest number allowed by seed
        seed = int(hsh.hexdigest(), 16) % 4294967295
        np.random.seed(seed)
        rand_indices = np.random.permutation(
            self.config['dimensionality'])[:self.config['num_indices']]
        pos_indices = rand_indices[rand_indices.size // 2:]
        neg_indices = rand_indices[:rand_indices.size // 2]
        unknown_vec = np.zeros((1, self.config['dimensionality']))
        unknown_vec[0, pos_indices] = 1
        unknown_vec[0, neg_indices] = -1
        unknown_vec = IndexMatrix(unknown_vec,
                                  ['$UNKNOWN$'],
                                  list(range(self.config['dimensionality'])))
        return unknown_vec / unknown_vec.norm()

    def identify(self, sentence):
        """
        Create a sentence vector for each language.
        When a word is unknown for the given language, it is treated as $UNKNOWN$.
        """
        words = sentence.split(" ")
        best_lang = None
        best_score = 0
        assure_consistency = self.config.get('assure_consistency', False)
        for language, mat in self.matrix.items():
            distance = 0
            for w in words:
                if w in mat.row2word:
                    wordvec = mat[w]
                    distance += abs(pydsm.similarity.cos(wordvec,
                                                         self.langvectors[language],
                                                         assure_consistency=assure_consistency)[0, 0])

            if distance > best_score:
                best_lang = language
                best_score = distance

        return best_lang

    def train(self, corpora):
        """
        Train the model according to the class documentation.
        """
        self.matrix = {}
        for language, corpus in corpora.items():
            print("Reading {}...".format(language))
            self.matrix[language] = self.build(corpus)
            langmodel = self.matrix[language].sum(axis=0)
            langmodel.row2word = [language]
            self.langvectors = self.langvectors.merge(langmodel)

    def build(self, text):
        """
        Create a random indexing space for each language.
        """
        model = RandomIndexing(corpus=text, config=self.config).matrix
        return model
