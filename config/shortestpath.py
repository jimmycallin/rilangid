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
        'description': "This gives the best results for shortest path.",
        'rimodel': rimodel,
        'assert_clean_repo': True,
    }
    configs = [config]

    return configs
