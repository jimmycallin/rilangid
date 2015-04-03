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
        Todo.
        """
        super().__init__(config=config)


class LetterBased(RILangID):

    """
    Todo.
    """

    @classmethod
    def words_to_blocks(cls, corpus, block_size):
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
        Todo.
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
            self.store(self.config['store_path'])
            print("Stored model at {}".format(self.config['store_path']))

        del self.config['language']

        print("Done training model.")

    def build(self, text):
        return IndexMatrix(*self._build(self.words_to_blocks(text, self.config['block_size'])))


# # # # # # Implemented models # # # # # #

class RILangVectorAddition(LetterBased):

    """
    Todo.
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


class RILangVectorPermutation(LetterBased):

    """
    Todo.
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

        row2word = [self.config['language']]
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorConvolution(LetterBased):

    """
    Todo.
    """

    def _build(self, block_stream):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            #print("Block {}: {}".format('init', block_vector))
            for k, char in enumerate(block):
                block_vector = np.multiply(
                    block_vector, self.get_index_vector(char))
                block_vector = np.roll(block_vector, 1)
                #print("Block {}: {}".format(k, block_vector))

            text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorNgrams(LetterBased):

    """
    Todo.
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
    Todo.
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

    @DSM.matrix.setter
    def matrix(self, mat):
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

    def __init__(self, config=None):
        super().__init__(config=config)

    def identify(self, sentence):
        words = sentence.split()
        best_lang = None
        best_score = 0

        # If sentence is only one word, take the language with the highest norm
        # of word vector.
        # if len(words) == 1:
        #     best_score = 0
        #     best_lang = None
        #     for language, langmat in self.matrix.items():
        #         if words[0] in langmat.word2row:
        #             norm = langmat[words[0]].norm()
        #             if norm > best_score:
        #                 best_score = norm
        #                 best_lang = language
        #     return best_lang

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
        self.store(self.config['store_path'])
        print("Stored model at {}".format(self.config['store_path']))

    def build(self, text):
        return RandomIndexing(corpus=text, config=self.config).matrix


class Eigenvectors(RILangID):

    def __init__(self, config=None):
        super().__init__(config=config)
        self.langvectors = IndexMatrix({})
        self.unknown_vec = self.create_unknown()

    def create_unknown(self):
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
        unknown_vec = IndexMatrix(
            unknown_vec, ['$UNKNOWN$'], list(range(self.config['dimensionality'])))
        return unknown_vec / unknown_vec.norm()

    def identify(self, sentence):
        words = sentence.split(" ")
        best_lang = None
        best_score = 0
        for language, mat in self.matrix.items():
            distance = 0
            knowns, unknowns = [], []
            for w in words:
                if w in mat.row2word:
                    wordvec = mat[w]
                    distance += abs(pydsm.similarity.cos(wordvec,
                                                         self.langvectors[
                                                             language],
                                                         assure_consistency=self.config.get('assure_consistency', False))[0, 0])

            if distance > best_score:
                best_lang = language
                best_score = distance

        return best_lang

    def train(self, corpora):
        self.matrix = {}
        for language, corpus in corpora.items():
            print("Reading {}...".format(language))
            self.matrix[language] = self.build(corpus)
            langmodel = self.matrix[language].sum(axis=0)
            langmodel.row2word = [language]
            self.langvectors = self.langvectors.merge(langmodel)
        self.store(self.config['store_path'])
        print("Stored model at {}".format(self.config['store_path']))

    def build(self, text):
        model = RandomIndexing(corpus=text, config=self.config).matrix
        return model
