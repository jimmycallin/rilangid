from models import *
from evaluate import languages

def get_configs():
    rimodel = MultipleNgrams
    block_size = 8

    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': block_size,
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'tags': ['multiple ngrams'],
        'description': "Testing multiple ngram sizes.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/multingram.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'assert_clean_repo': False
    }

    return [config]