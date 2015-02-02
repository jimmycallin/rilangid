from pydsm import IndexMatrix
from pydsm.model import DSM
import scipy
import numpy as np
import hashlib

def words_to_chars(corpus):
    "Converts list of words to list of characters"
    for sentence in corpus:
        for char in sentence:
            yield char

def words_to_blocks(corpus, block_size):
    block = []
    for char in words_to_chars(corpus):
        if len(block) < block_size:
            block.append(char)
            continue
        yield block
        block = block[1:] + [char]
    yield block

class RILangID(DSM):
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

        super(RILangID, self).__init__(matrix,
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
                                       directed=directed)
        
        if corpus:
            self.matrix = IndexMatrix(*self._build(words_to_blocks(corpus, window_size[0])))

    
    def get_index_vector(self, context):
        if context not in self.word_vectors:
            # Create index vector if not exist
            hsh = hashlib.md5()
            hsh.update(context.encode())
            seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
            np.random.seed(seed)
            rand_indices = np.random.permutation(self.config['dimensionality'])[:self.config['num_indices']]
            pos_indices = rand_indices[rand_indices.size // 2:]
            neg_indices = rand_indices[:rand_indices.size // 2]
            index_vector = np.zeros((1,self.config['dimensionality']))
            index_vector[0,pos_indices] = 1
            index_vector[0,neg_indices] = -1
            self.word_vectors[context] = index_vector
        return self.word_vectors[context]

class RILangVectorAddition(RILangID):
    def _build(self, text):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """
        word_to_col = {}

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in text:
            block_vector = np.zeros_like(text_vector)
            for k, char in enumerate(block):
                block_vector += self.get_index_vector(char)

            text_vector += block_vector

        row2word = [self.config['language']] if 'language' in self.config else ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word

class RILangVectorPermutation(RILangID):
    def _build(self, block_stream):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """
        word_to_col = {}
        indices = np.arange(self.config['num_indices'])

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            for k, char in enumerate(block):
                char = "{}_{}".format(char, k)
                block_vector = np.multiply(block_vector,self.get_index_vector(char))

            text_vector += block_vector

        row2word = [self.config['language']]
        col2word = indices.tolist()

        return text_vector, row2word, col2word


class RILangVectorConvolution(RILangID):
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
                block_vector = np.multiply(block_vector,self.get_index_vector(char))
                block_vector = np.roll(block_vector, 1)
                #print("Block {}: {}".format(k, block_vector))

            text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']
        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word


class RILangVectorNgrams(RILangID):
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


class RILangVectorNgramsGramSchmidt(RILangVectorNgrams):
    @DSM.matrix.setter
    def matrix(self, mat):
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            self._matrix = mat
        else:
            orthogonalized = scipy.sparse.coo_matrix(self.Gram_Schmidt(mat.to_ndarray()))
            self._matrix = mat._new_instance(orthogonalized)
    

    @classmethod
    def Gram_Schmidt(cls, vecs, row_wise_storage=True):
        """
        Apply the Gram-Schmidt orthogonalization algorithm to a set
        of vectors. vecs is a two-dimensional array where the vectors
        are stored row-wise, or vecs may be a list of vectors, where
        each vector can be a list or a one-dimensional array.
        An array basis is returned, where basis[i,:] (row_wise_storage
        is True) or basis[:,i] (row_wise_storage is False) is the i-th
        orthonormal vector in the basis.

        This function does not handle null vectors, see Gram_Schmidt
        for a (slower) function that does.
        """
        from numpy.linalg import inv
        vecs = np.asarray(vecs)  # transform to array if list of vectors
        m, n = vecs.shape
        basis = np.array(np.transpose(vecs))
        eye = np.identity(n).astype(float)
        
        basis[:,0] /= np.sqrt(np.dot(basis[:,0], basis[:,0]))
        for i in range(1, m):
            v = basis[:,i]/np.sqrt(np.dot(basis[:,i], basis[:,i]))
            U = basis[:,:i]
            P = eye - np.dot(U, np.dot(inv(np.dot(np.transpose(U), U)), np.transpose(U)))
            basis[:, i] = np.dot(P, v)
            basis[:, i] /= np.sqrt(np.dot(basis[:, i], basis[:, i]))

        return np.transpose(basis) if row_wise_storage else basis



class ShortestPath(RILangID):
    def _build(self, block_stream):
        """
        Builds the text vector from text iterator.
        Returns: text vector, row ids, column ids.
        """

        text_vector = np.zeros((1, self.config['dimensionality']))
        for block in block_stream:
            block_vector = np.ones_like(text_vector)
            #print("Block {}: {}".format('init', block_vector))
            block_vector = np.multiply(block_vector,self.get_index_vector("".join(block)))
            block_vector = np.roll(block_vector, 1)

            text_vector += block_vector

        if 'language' in self.config:
            row2word = [self.config['language']]
        else:
            row2word = ['']

        col2word = list(range(self.config['dimensionality']))

        return text_vector, row2word, col2word