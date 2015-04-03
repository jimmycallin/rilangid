# rilangid
Language identification using Random Indexing

This is the result of a project course in Uppsala University where I tested language identification using Random Indexing models.

A number of different candidate models where tested, and are to be described below. The best one was the ShortestPath model, which produced an F-score of 99.43% given the training data.

## Requirements for running

The experiments are built upon [PyDSM](https://github.com/jimmycallin/expy), a library made for exploring distributional semantic models. Make sure you have this installed first.

You also need to have [Expy](https://github.com/jimmycallin/expy) installed. This is for saving your experiment results as well as calculating result measures.

Everything is tested on Python 3.4.

## Running the experiments

When you are sure you have the necessary libraries installed, you should be able to reproduce the results by typing:

    python evaluate.py config.shortestpath

This uses the configuration found in config/shortestpath.py when running the experiment:

    {'dimensionality': '2000',
     'directed': 'True',
     'num_indices': '8',
     'ordered': 'False',
     'rimodel': "<class 'models.ShortestPath'>",
     'test_path': '/Users/jimmy/dev/projects/rilangid/resources/test/reproduce/',
     'train': 'True',
     'window_size': '(100, 100)'}

 After a long while, you should get a result similar to this:

    Precision: 0.9968449289229714
    Recall: 0.991852487135506
    F-score: 0.9943254868891508