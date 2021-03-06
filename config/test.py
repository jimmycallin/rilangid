from models import *
from evaluate import languages
import expy
import pymysql
import dataset

cursor = pymysql.connect(user='root').cursor()
cursor.execute("drop database if exists testExpy; create database testExpy; use testExpy;" + expy.sql)
db = dataset.connect('mysql+pymysql://root@localhost/testExpy')
expy.project._db = db


def get_configs():
    rimodel = Eigenvectors

    config = {
        'project_name': 'RILangID',
        'languages': ['bul', 'lav'],
        'test_path': "/Users/jimmy/dev/projects/rilangid/resources/test/devsmall/",
        'window_size': (2,2),
        'dimensionality': 2000,
        'num_indices': 8,
        'directed': True,
        'ordered': False,
        'tags': ['test'],
        'description': "Just testing.",
        'store_path': '/Users/Jimmy/dev/projects/rilangid/resources/models/test.{}.dsm'.format(rimodel.__name__),
        'rimodel': rimodel,
        'train': True,
        'assert_clean_repo': False
    }

    return [config]