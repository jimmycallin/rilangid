from models import *
from evaluate import languages

def get_configs():
    rimodel = Eigenvectors
    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/resources/test/reproduce/",
        'window_size': (100,100),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'tags': ['eigenvectors'],
        'description': "Implementation of eigenvector algorithm.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/resources/models/eigenvector.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'assert_clean_repo': False,
        'train': False,
        'assure_consistency': False
    }
    return [config]