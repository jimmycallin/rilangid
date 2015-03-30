from models import *
from evaluate import languages

def get_configs():
    rimodel = ShortestPath

    config = {
        'project_name': 'RILangIDbigtest',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'window_size': (10,10),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': False,
        'tags': ['test'],
        'description': "Just testing.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/test.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'assert_clean_repo': False,
        'train': False
    }

    return [config]