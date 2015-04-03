from models import *
from evaluate import languages


def get_configs():
    rimodel = ShortestPath
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'window_size': (100, 100),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'tags': ['shortestpath'],
        'description': "This gives the best results for shortest path, without special case for single words.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/shortestpath.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'assert_clean_repo': True,
        'train': False
    }
    configs = [config]

    return configs
