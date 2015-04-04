from models import *
from evaluate import languages
import itertools

def get_configs():

    configs = []

    rimodel = RILangVectorConvolution
    config = {
        'project_name': 'RILangID.final2',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/resources/test/reproduce/",
        'block_size': 3,
        'dimensionality': 10000,
        'num_indices': 10000,
        'directed': True,
        'ordered': True,
        'description': "Reproducing exact results as in paper.",
        'tags': ['letterbased', 'ngrams', 'convolution'],
        'assert_clean_repo': False,
        'rimodel': rimodel,
    }
    configs.append(config.copy())



    rimodel = ShortestPath
    config = {
        'project_name': 'RILangID.final2',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/resources/test/reproduce/",
        'window_size': (100, 100),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'tags': ['shortestpath'],
        'description': "Implementation of shortest path algorithm, testing various versions of directed and ordered context.",
        'rimodel': rimodel,
        'assert_clean_repo': False,
    }

    return configs