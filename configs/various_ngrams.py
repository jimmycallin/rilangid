from models import *
from evaluate import languages

def get_configs():
    rimodel = RILangVectorNgrams
    block_size = 3

    config = {
        'project_name': 'RILangID',
        'languages': languages,
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': block_size,
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': True,
        'description': "Testing various values on block sizes",
        'rimodel': rimodel
    }

    configs = []
    block_sizes = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    for block_size in block_sizes:
        conf = config.copy()
        conf['block_size'] = block_size
        conf['store_path'] = '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/block_size.{}.{}.dsm'.format(rimodel.__name__, block_size)
        configs.append(conf)

    return configs