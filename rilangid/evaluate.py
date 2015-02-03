from models import *
import os
import re
import pyresult
import pydsm
import sys
from importlib import import_module
from pprint import pprint

languages = ['eng', 'bul', 'nld', 'fin', 'lav', 'lit', 'ron', 'deu', 'por', 'slk', 'slv', 'ita', 'spa', 'pol', 'hun', 'ell', 'dan', 'fra', 'est', 'ces', 'afr', 'swe']

def load_project(name, test_sentences, assert_clean_repo=True):
    try:
        proj = pyresult.Project.get_project(name=name, assert_clean_repo=assert_clean_repo)
        print("Loaded existing project {}.".format(name))
    except KeyError:
        proj = pyresult.Project(name=name,
                                test_data=test_sentences, assert_clean_repo=assert_clean_repo)
        print("Created new project {}.".format(name))
    return proj

def get_corpora(languages):
    corpora = {}
    for language in languages:
        corpora[language] = open('resources/train/reproduce/{}.txt'.format(language))
    return corpora

def load_test_sentences(path):
    sentences = {}
    path_content = os.listdir(path)
    for f in path_content:
        if re.match('(\w+)_\d+_p.txt', f):
            with open(path + f) as content:
                lang = re.findall('(\w+)_\d+_p.txt$', f)[0]
                sentences[content.readline().strip()] = lang

    return sentences

def evaluate(model_path, test_sentences):
    model = pydsm.load(model_path)
    pred_results = {}
    for sentence, language in test_sentences.items():
        pred = model.identify(sentence)
        pred = pred.row2word[0]
        pred_results[sentence] = pred
        if pred == language:
            print("Correct! {} == {}".format(pred, language))
        else:
            print("Incorrect. {} != {}".format(pred, language))
    return pred_results

def run_experiment(configuration):
    test_sentences = load_test_sentences(configuration['test_path'])
    proj = load_project(configuration['project_name'], test_sentences, assert_clean_repo=configuration.get('assert_clean_repo', True))
    corpora = get_corpora(configuration['languages'])

    print("Starting training with configuration:")
    pprint(configuration, width=80)
    model = configuration['rimodel'](config=configuration)
    model.train(corpora)
    print("Evaluating model...")
    
    sentence_results = evaluate(configuration['store_path'], test_sentences)
    experiment = proj.new_experiment(predicted=sentence_results,
                                     configuration=configuration,
                                     description=configuration['description'])

    experiment.experiment_report(f1_score=True,
                                 precision=True,
                                 recall=True)


if __name__ == "__main__":
    configs = import_module(sys.argv[1]).get_configs()
    for conf in configs:
        run_experiment(conf)