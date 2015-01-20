def build_language_vector(corpus, language, model):
    language_vector = type(model)(corpus=corpus,
                                  window_size=model.config['window_size'],
                                  dimensionality=model.config[
                                      'dimensionality'],
                                  num_indices=model.config['num_indices'],
                                  directed=model.config['directed'],
                                  ordered=model.config['ordered'],
                                  language=language)
    return language_vector

def train(corpora, config):
    print("Creating new model {}.".format(config['rimodel'].__name__))
    model = config['rimodel'](corpus=None,
                              window_size=(config['block_size'], 0),
                              dimensionality=config['dimensionality'],
                              num_indices=config['num_indices'],
                              directed=config['directed'],
                              ordered=config['ordered'])

    for i, (language, corpus) in enumerate(corpora.items()):
        if hasattr(model, 'word2row') and language in model.word2row:
            print("{} already created, skipping.".format(language))
            continue
        print("Building {} vector... {} / {}".format(language, i + 1, len(corpora)))
        
        lang_vector = build_language_vector(corpus, language, model)
        model.matrix = model.matrix.merge(lang_vector.matrix)
        model.store(config['store_path'])
        print("Stored {} vector at {}".format(language, config['store_path']))

def identify(model, sentence):
    text_model = type(model)(corpus=sentence,
                             window_size=model.config['window_size'],
                             dimensionality=model.config['dimensionality'],
                             num_indices=model.config['num_indices'],
                             directed=model.config['directed'],
                             ordered=model.config['ordered'])

    return model.nearest_neighbors(text_model.matrix)