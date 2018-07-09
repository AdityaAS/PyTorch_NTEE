# Entry point into the code. Each function in this file can be called as python cli.py <methodname> <arg1> <arg2> etc.
import click
import subprocess
import logging
import multiprocessing
import pickle
from sklearn.cross_validation import train_test_split

import os
import re
import pdb
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr

from ntee.utils.model_reader import ModelReader
from ntee.utils.model_reader import PyTorchModelReader
from ntee.sentence_similarity import eval_sick, eval_sts
from configparser import ConfigParser

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

from ntee.utils.my_tokenizer import RegexpTokenizer
from ntee.utils.abstract_db import AbstractDB
from ntee.utils.vocab import Vocab
import numpy as np

from ntee.utils import word2vec
from ntee.utils.entity_db import EntityDB
import torch
from torch import nn
import train_ntee

@cli.command()
@click.argument('out_dir', type=click.Path(exists=True, file_okay=False))
def download_dbpedia_abstract_files(out_dir):
    for n in range(114):
        url = 'https://s3-ap-northeast-1.amazonaws.com/ntee/pub/dbpedia_abstract/abstracts_en%d.ttl.gz' % (n,)
        click.echo('Getting %s' % url)
        subprocess.getoutput('wget -P %s/ %s' % (out_dir, url))

@cli.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--white-list', type=click.Path(), default=None)
def build_abstract_db(**kwargs):
    AbstractDB.build(**kwargs)

@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def build_entity_db(dump_file, out_file, **kwargs):
    db = EntityDB.build(dump_file, **kwargs)
    db.save(out_file)


@cli.command()
@click.argument('out_file', type=click.Path())
@click.option('--config', default="./ntee/data.cfg")
@click.option('--min-word-count', default=5)
@click.option('--min-entity-count', default=3)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--white-list', default=None, type=click.Path())
@click.option('--chunk-size', default=30)
def build_vocab(out_file, config, **kwargs):
    cfg = ConfigParser()
    cfg.read(config)
    abstractdb = pickle.load(open(cfg.get('DEFAULT', 'abstractdb'), 'rb'))
    entitydb = EntityDB.load(cfg.get('DEFAULT', 'entitydb'))

    vocab = Vocab.build(abstractdb, entitydb, **kwargs)
    vocab.save('./data/vocabs/' + out_file)


@cli.group(name='word2vec')
def word2vec_group():
    pass

@word2vec_group.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--learn-entity/--no-entity', default=True)
@click.option('--abstract-db', type=click.Path(), default=None)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def generate_corpus(dump_file, entity_db_file, out_file, abstract_db, **kwargs):
    entity_db = EntityDB.load(entity_db_file)
    if abstract_db:
        abstract_db = AbstractDB.load(abstract_db)

    word2vec.generate_corpus(dump_file, entity_db, out_file, abstract_db, **kwargs)

@word2vec_group.command(name='train')
@click.argument('corpus_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--mode', type=click.Choice(['sg', 'cbow']), default='sg')
@click.option('--dim-size', default=300)
@click.option('--window', default=10)
@click.option('--min-count', default=3)
@click.option('--negative', default=5)
@click.option('--epoch', default=5)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def train_word2vec(corpus_file, out_file, **kwargs):
    word2vec.train(corpus_file, out_file, **kwargs)

@cli.command()
@click.option('--config', default="./ntee/data.cfg")
@click.option('--modelname', default="NTEE")
@click.option('--w2vec', type=str, default='SG')
@click.option('--mode', type=click.Choice(['paragraph', 'sentence']), default='paragraph')
@click.option('--text-len', default=2000)
@click.option('--dim-size', default=50)
@click.option('--negative', default=30)
@click.option('--epoch', default=1)
@click.option('--batch-size', default=200)
@click.option('--word-static', is_flag=True)
@click.option('--entity-static', is_flag=True)
@click.option('--include-title/--no-title', default=True)
@click.option('--optimizer', default='adam')
@click.option('--lr', default=0.1)
@click.option('--dev-size', default=1000)
@click.option('--patience', default=1)
@click.option('--num-links', type=int)
@click.option('--random-seed', default=0)
@click.option('--gpu', default=7)
def train_ntee_model(config, modelname, w2vec,  **kwargs):
    cfg = ConfigParser()
    cfg.read(config)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(kwargs['gpu'])
    out_file = './results/models/' + modelname + '_dim=' + str(kwargs['dim_size']) + '_mode=' + str(kwargs['mode']) + '_w2vec=' + str(w2vec)
    kwargs['out_file'] = out_file

    db_file = cfg.get('DEFAULT', 'abstractdb')
    entity_db_file = cfg.get('DEFAULT', 'entitydb')
    vocab_file = cfg.get('NTEE', 'vocabdb')

    db = AbstractDB.load(db_file)
    entity_db = EntityDB.load(entity_db_file)
    vocab = Vocab.load(vocab_file)
    word2vec = cfg.get(w2vec, 'w2vec_' + str(kwargs['dim_size']))

    if word2vec:
        w2vec = ModelReader(word2vec)
    else:
        w2vec = None

    train_ntee.train_ntee(db, entity_db, vocab, w2vec, **kwargs)

@cli.command()
@click.argument('model_file', type=click.Path())
@click.option('--config', default="./ntee/data.cfg")
def sick(model_file, config):
    cfg = ConfigParser()
    cfg.read(config)

    dataset_file = './data/STS/SICK.txt'
    out_file = './results/evaluation/' + model_file.split('/')[-1] + '.sick'
    modelname = (model_file.split('/')[-1]).split('_')[0]
    vocab_file = cfg.get(modelname, 'vocabdb')

    eval_sick(model_file, vocab_file, dataset_file, modelname, out_file)

@cli.command()
@click.argument('model_file', type=click.Path())
@click.option('--config', default="./ntee/data.cfg")
def sts(model_file, config):
    cfg = ConfigParser()
    cfg.read(config)

    dataset_dir = './data/STS/sts-en-test-gs-2014/'
    out_file = './results/evaluation/' + model_file.split('/')[-1] + '.stseval'
    modelname = model_file.split('/')[-1].split('_')[0]
    vocab_file = cfg.get(modelname, 'vocabdb')

    eval_sts(model_file, vocab_file, dataset_dir, modelname, out_file)

def main():
    cli()

if __name__ =="__main__":
    main()
