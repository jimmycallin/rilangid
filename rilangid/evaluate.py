from models import *
import rilangid
import pyresult

base_description = """
http://arxiv.org/abs/1412.7026

The main algorithm is basically elementwise multiplication of permuted character n-gram vectors of a sentence. The character ngram vectors (blocks) are added together to form a text vector, which is later compared to language vectors that are formed in the same manner, but on more data.

Their best results were retrieved by using: Window size: 3+0
Dimensionality: 10 000
Num 1/-1: 10 000 (5k each)
Composed by permuted elementwise multiplication through convolution.
"""

def load_project(name, test_sentences, base_configuration, description):
    try:
        proj = pyresult.Project.get_project(name=name)
        print("Loaded existing project {}.".format(name))
    except KeyError:
        proj = pyresult.Project(name=name,
                                test_data=test_sentences,
                                base_configuration=base_configuration,
                                description=description)
        print("Created new project {}.".format(name))
    return proj

base_configuration = {
    'languages': rilangid.lang_map.values(),
    'test_path': "/Users/jimmy/dev/projects/rilangid/rilangid/resources/test/reproduce/"
}

config = {
    'block_size': 3,
    'dimensionality': 10000,
    'num_indices': 10000,
    'directed': True,
    'ordered': True
}

rimodels = [RILangVectorConvolution, RILangVectorConvolutionNgrams, RILangVectorAddition, RILangVectorPermutation]
descriptions = ["Reproducing results from article.",
                "See if character ngrams do just as well as character composing.",
                "Make sure simple addition of vectors actually is bad.",
                "Make sure convolution and permutation are the same."]

test_sentences = rilangid.load_test_sentences(base_configuration['test_path'])
proj = load_project("RILangID", test_sentences, base_configuration, base_description)
for descr, rimodel in zip(descriptions, rimodels):
    config['rimodel'] = rimodel
    config['store_path'] = '/Users/Jimmy/dev/projects/rilangid/rilangid/resources/reproduce.{}.{}.dsm'.format(rimodel.__name__, config['block_size'])
    print("Starting training...")
    rilangid.train(base_configuration['languages'], config)
    print("Evaluating model...")
    sentence_results = rilangid.evaluate(config['store_path'], test_sentences)
    experiment = proj.new_experiment(predicted=sentence_results,
                                     configuration=config,
                                     description=descr)

    experiment.experiment_report(f1_score=True,
                                 precision=True,
                                 recall=True)


print("Done!")