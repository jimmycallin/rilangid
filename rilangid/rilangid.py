import pydsm
import plainstream
import os
import numpy as np
import sys
import re
from collections import defaultdict
from pydsm.model import DSM
from models import *
from sklearn.metrics import f1_score

lang_map = {'af': 'afr', 'bg': 'bul', 'cs': 'ces', 'da': 'dan', 'nl': 'nld', 'de': 'deu', 'en': 'eng', 'et': 'est', 'fi': 'fin', 'fr': 'fra', 'el':
            'ell', 'hu': 'hun', 'it': 'ita', 'lv': 'lav', 'lt': 'lit', 'pl': 'pol', 'pt': 'por', 'ro': 'ron', 'sk': 'slk', 'sl': 'slv', 'es': 'spa', 'sv': 'swe'}

config = {}

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
                                  language=language)
    return language_vector


def train(languages, conf):
    global config
    config = conf
    print("Creating new model {}.".format(config['rimodel'].__name__))
    model = config['rimodel'](corpus=None,
                    window_size=(config['block_size'], 0),
                    dimensionality=config['dimensionality'],
                    num_indices=config['num_indices'],
                    directed=config['directed'],
                    ordered=config['ordered'])

    for i, language in enumerate(languages):
        if hasattr(model, 'word2row') and language in model.word2row:
            print("{} already created, skipping.".format(language))
            continue
        print("Building {} vector... {} of {} languages.".format(language,
                                                                 i + 1,
                                                                 len(languages)))
        lang_vector = build_language_vector(get_corpus(language),
                                            language,
                                            model)

        model.matrix = model.matrix.merge(lang_vector.matrix)
        model.store(config['store_path'])
        print("Stored {} vector at {}".format(language, config['store_path']))


def identify(model, sentence):
    sentence_to_chars = chars_to_blocks(words_to_chars([sentence]),
                                        config['block_size'])
    text_model = type(model)(corpus=sentence_to_chars,
                             window_size=model.config['window_size'],
                             dimensionality=model.config['dimensionality'],
                             num_indices=model.config['num_indices'],
                             directed=model.config['directed'],
                             ordered=model.config['ordered'],
                             language=sentence,)

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
    for sentence, language in test_sentences.items():
        pred = identify(model, sentence)
        pred = pred.row2word[0]
        pred_results[sentence] = pred
        if pred == language:
            print("Correct! {} == {}".format(pred, language))
        else:
            print("Incorrect. {} != {}".format(pred, language))
    return pred_results