import pydsm
import pyresult
import plainstream
import os
import numpy as np
import sys
import re
from collections import defaultdict
from pydsm.model import DSM
from models import RILangVectorPermutation, RILangVectorAddition, RILangVectorConvolution
from sklearn.metrics import f1_score

base_description = """
http://arxiv.org/abs/1412.7026

The main algorithm is basically elementwise multiplication of permuted character n-gram vectors of a sentence. The character ngram vectors (blocks) are added together to form a text vector, which is later compared to language vectors that are formed in the same manner, but on more data.

Their best results were retrieved by using: Window size: 3+0
Dimensionality: 10 000
Num 1/-1: 10 000 (5k each)
Composed by permuted elementwise multiplication through convolution.
"""

lang_map = {'af': 'afr', 'bg': 'bul', 'cs': 'ces', 'da': 'dan', 'nl': 'nld', 'de': 'deu', 'en': 'eng', 'et': 'est', 'fi': 'fin', 'fr': 'fra', 'el':
            'ell', 'hu': 'hun', 'it': 'ita', 'lv': 'lav', 'lt': 'lit', 'pl': 'pol', 'pt': 'por', 'ro': 'ron', 'sk': 'slk', 'sl': 'slv', 'es': 'spa', 'sv': 'swe'}


def words_to_chars(corpus):
    "Converts list of words to list of characters"
    for sentence in corpus:
        for char in sentence:
            yield char


def chars_to_blocks(chars, block_size):
    block = []
    for char in chars:
        if len(block) < block_size:
            block.append(char)
            continue
        yield block
        block = block[1:] + [char]
    yield block


def get_corpus(language):
    text = open('resources/train/reproduce/{}.txt'.format(language))
    return chars_to_blocks(words_to_chars(text), config['block_size'])


def build_language_vector(corpus, language, model):
    language_vector = type(model)(corpus=corpus,
                                  window_size=model.config['window_size'],
                                  dimensionality=model.config[
                                      'dimensionality'],
                                  num_indices=model.config['num_indices'],
                                  directed=model.config['directed'],
                                  ordered=model.config['ordered'],
                                  language=language,
                                  word_vectors=model.word_vectors)
    return language_vector


def train(languages, model):
    for i, language in enumerate(languages):
        if language in model.word2row:
            print("{} already created, skipping.".format(language))
            continue
        print("Building {} vector... {} of {} languages.".format(language,
                                                                 i + 1,
                                                                 len(languages)))
        lang_vector = build_language_vector(get_corpus(language),
                                            language,
                                            model)

        model.matrix = model.matrix.merge(lang_vector.matrix)
        model.word_vectors.update(lang_vector.word_vectors)
        model.store(config['store_path'])
        print("Stored {} vector at {}".format(language, config['store_path']))


def identify(model, sentence, word2col):
    sentence_to_chars = chars_to_blocks(words_to_chars([sentence]),
                                        config['block_size'])
    text_model = type(model)(corpus=sentence_to_chars,
                             window_size=model.config['window_size'],
                             dimensionality=model.config['dimensionality'],
                             num_indices=model.config['num_indices'],
                             directed=model.config['directed'],
                             ordered=model.config['ordered'],
                             language=sentence,
                             word_vectors=model.word_vectors)

    text_vector = text_model.matrix
    return model.nearest_neighbors(text_vector)


def load_test_sentences(path):
    sentences = {}
    path_content = os.listdir(path)
    for f in path_content:
        if re.match('(\w+)_\d+_p.txt', f):
            with open(path + f) as content:
                lang = lang_map[re.findall('(\w+)_\d+_p.txt$', f)[0]]
                sentences[content.readline().strip()] = lang

    return sentences


def evaluate(model_path, test_sentences):
    model = pydsm.load(model_path)
    pred_results = {}
    word2col = model.word_vectors
    for sentence, language in test_sentences.items():
        pred = identify(model, sentence, word2col)
        pred = pred.row2word[0]
        pred_results[sentence] = pred
        if pred == language:
            print("Correct! {} == {}".format(pred, language))
        else:
            print("Incorrect. {} != {}".format(pred, language))
    return pred_results


rimodel = RILangVectorPermutation
block_size = 3
base_configuration = {
    'languages': lang_map.values(),
    'test_path': "resources/test/reproduce/"
}
config = {
    'block_size': block_size,
    'rimodel': rimodel,
    'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/reproduce.{}.{}.dsm'.format(rimodel.__name__, block_size),
    'dimensionality': 10000,
    'num_indices': 10000,
    'directed': True,
    'ordered': True
}


def load_project(name, test_data, base_configuration, description):
    try:
        proj = pyresult.Project.get_project(name=name)
    except KeyError:
        proj = pyresult.Project(name=name,
                                test_data=test_sentences,
                                base_configuration=config,
                                description=base_description)
    return proj


def run():

    if os.path.exists(config['store_path']):
        print("Loading already created model.")
        model = pydsm.load(config['store_path'])
    else:
        print("Creating new model.")
        model = rimodel(corpus=None,
                        matrix=pydsm.IndexMatrix({}),
                        window_size=(config['block_size'], 0),
                        dimensionality=config['dimensionality'],
                        num_indices=config['num_indices'],
                        directed=config['directed'],
                        ordered=config['ordered'],
                        is_ngrams=False)

    print("Starting training...")
    train(base_configuration['languages'], model)
    print("Done!")

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        run()
    elif sys.argv[1] == 'test':
        test_sentences = load_test_sentences(base_configuration['test_path'])
        description = sys.argv[3]
        proj = load_project(
            "RILangID", test_sentences, base_configuration, description)
        sentence_results = evaluate(sys.argv[2], test_sentences)
        experiment = proj.new_experiment(predicted=sentence_results,
                                         configuration=config,
                                         description="First repro")

        experiment.experiment_report(f1_score=True,
                                     precision=True,
                                     recall=True)
