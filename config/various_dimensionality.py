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
        'description': "Testing various values on dimensionality",
        'rimodel': rimodel,
        'tags': ['dimensionality']
    }

    configs = []
    #dimensionality = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16385)
    dimensionality = (32770, 65540, 131080)
    for dimensions in dimensionality:
        conf = config.copy()
        conf['dimensionality'] = dimensions
        conf['num_indices'] = dimensions
        conf['store_path'] = '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/models/dimensionality.{}.{}.dsm'.format(rimodel.__name__, dimensions)
        configs.append(conf)

    return configs