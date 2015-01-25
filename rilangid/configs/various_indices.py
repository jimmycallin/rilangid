from models import *
from evaluate import lang_map

def get_configs():
    rimodel = RILangVectorNgrams
    block_size = 3

    config = {
        'project_name': 'RILangID',
        'languages': lang_map.values(),
        'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/",
        'block_size': block_size,
        'dimensionality': 2000,
        'num_indices': 2000,
        'directed': True,
        'ordered': True,
        'description': "Testing various values on indices (fixed bug)",
        'rimodel': rimodel
    }

    configs = []
    num_indices = (8, 16, 32, 64, 128, 256, 512, 1024, 1999)
    for indices in num_indices:
        conf = config.copy()
        conf['num_indices'] = indices
        conf['store_path'] = '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/num_indices.{}.{}.dsm'.format(rimodel.__name__, indices)
        configs.append(conf)

    return configs