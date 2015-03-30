from models import *
from evaluate import languages
import itertools

def get_configs():

    configs = []
    
    rimodel = RILangVectorNgrams
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': 3,
        'dimensionality': 10000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'description': "Showing that sparse permutation works just as well.",
        'tags': ['letterbased', 'ngrams'],
        'rimodel': rimodel,
        'assert_clean_repo': True,
        'train': True,
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/block_size.{}.{}.dsm'.format(rimodel.__name__, 3)
    }
    #configs.append(config.copy())

    rimodel = RILangVectorNgrams
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': 3,
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'description': "Testing difference in dimensionality.",
        'tags': ['letterbased', 'ngrams'],
        'rimodel': rimodel,
        'assert_clean_repo': True,
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/block_size.{}.{}.dsm'.format(rimodel.__name__, 3)
    }
    #configs.append(config.copy())

    rimodel = RILangVectorPermutation
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': 3,
        'dimensionality': 10000,
        'num_indices': 10000,
        'directed': True,
        'ordered': True,
        'description': "Showing that permutation is just as good as convolution.",
        'tags': ['letterbased', 'ngrams', 'permutation'],
        'rimodel': rimodel,
        'assert_clean_repo': True,
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/block_size.{}.{}.dsm'.format(rimodel.__name__, 3)
    }    
    configs.append(config.copy())

    rimodel = RILangVectorConvolution
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': 3,
        'dimensionality': 10000,
        'num_indices': 10000,
        'directed': True,
        'ordered': True,
        'description': "Reproducing exact results as in paper.",
        'tags': ['letterbased', 'ngrams', 'convolution'],
        'assert_clean_repo': True,
        'rimodel': rimodel,
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/block_size.{}.{}.dsm'.format(rimodel.__name__, 3)
    }    
    configs.append(config.copy())

    rimodel = ShortestPath
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'window_size': (100,100),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'tags': ['shortestpath'],
        'description': "Implementation of shortest path algorithm, testing various versions of directed and ordered context.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/shortestpath.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'assert_clean_repo': True,
        'train': True
    }

    for directed, ordered in itertools.product((True, False), (True, False)):
        config = config.copy()
        config['directed'] = directed
        config['ordered'] = ordered
        configs.append(config)

    return configs