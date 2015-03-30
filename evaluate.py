from models import *
import os
import re
import expy
import pydsm
import sys
from importlib import import_module
from pprint import pprint

languages = ['eng', 'bul', 'nld', 'fin', 'lav', 'lit', 'ron', 'deu', 'por', 'slk', 'slv', 'ita', 'spa', 'pol', 'hun', 'ell', 'dan', 'fra', 'est', 'ces', 'afr', 'swe']

def load_project(configuration, assert_clean_repo=False):
    project_name = configuration.pop('project_name')
    test_sentences = load_test_sentences(configuration)
    proj = expy.Project.search_project(project_name=project_name, assert_clean_repo=assert_clean_repo)
    if proj:
        print("Loaded existing project {}.".format(project_name))
    else:
        proj = expy.Project(project_name=project_name, test_data=test_sentences, assert_clean_repo=assert_clean_repo)
        print("Created new project {}.".format(project_name))
    return proj

def get_corpora(languages):
    corpora = {}
    for language in languages:
        corpora[language] = open('resources/train/reproduce/{}.txt'.format(language))
    return corpora

def load_test_sentences(configuration):
    sentences = {}
    path_content = os.listdir(configuration['test_path'])
    for f in path_content:
        if re.match('(\w+)_\d+_p.txt', f):
            lang = re.findall('(\w+)_\d+_p.txt$', f)[0]
            if lang in configuration['languages']:
                with open(configuration['test_path'] + f) as content:
                    sentences[content.readline().strip()] = lang

    return sentences

def evaluate(model_path, test_sentences):
    model = pydsm.load(model_path)
    pred_results = {}
    for i, (sentence, language) in enumerate(test_sentences.items()):
        pred = model.identify(sentence)
        pred_results[sentence] = pred
        if pred == language:
            print("Correct! {} == {}. {} / {}".format(pred, language, i+1, len(test_sentences)))
        else:
            print("Incorrect. {} != {}. {} / {}".format(pred, language, i+1, len(test_sentences)))
    return pred_results

def run_experiment(configuration):
    proj = load_project(configuration, assert_clean_repo=configuration.pop('assert_clean_repo', True))
    test_sentences = proj.test_data
    corpora = get_corpora(configuration['languages'])

    print("Configuration:")
    pprint(configuration, width=80)
    if configuration.get('train', True):
        print("Starting training...")
        model = configuration['rimodel'](config=configuration)
        model.train(corpora)
    print("Evaluating model...")
    
    sentence_results = evaluate(configuration['store_path'], test_sentences)
    experiment = proj.new_experiment(predicted=sentence_results,
                                     tags=configuration.pop('tags', None),
                                     configuration=configuration,
                                     description=configuration['description'])

    experiment.experiment_report(f1_score=True,
                                 precision=True,
                                 recall=True)


if __name__ == "__main__":
    configs = import_module(sys.argv[1]).get_configs()
    for conf in configs:
        run_experiment(conf)