# -*- coding: utf-8 -*-

import click
import gzip
import os
import rdflib
import re
import urllib

from collections import defaultdict
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from ntee.utils.my_tokenizer import RegexpTokenizer
import pdb
import pickle
import json

def normalize(title):
    return title.replace('_', ' ').lower()


class AbstractDB():

    dbpedia_dict = {}

    def __init__(self, out_file):
        self._out_file = out_file

    @staticmethod
    def load(infile):
        return pickle.load(open(infile, 'rb'))
        
    @staticmethod
    def build(in_dir, out_file, pool_size, white_list):      
        target_files = [f for f in sorted(os.listdir(in_dir)) if f.endswith('ttl.gz')]

        if white_list is None:
            with closing(Pool(pool_size)) as pool:
                f = partial(_process_file, in_dir=in_dir)
                for ret in pool.imap(f, target_files):
                    for (key, obj) in ret:
                        AbstractDB.dbpedia_dict[key] = obj
        else:
            click.echo("Entity white list file%s"%white_list) 
            white_list = json.load(open(white_list, 'r'))
            with closing(Pool(pool_size)) as pool:
                f = partial(_process_file, in_dir=in_dir)
                for ret in pool.imap(f, target_files):
                    for (key, obj) in ret:
                        if key in white_list:
                            AbstractDB.dbpedia_dict[key] = obj

        pickle.dump(AbstractDB.dbpedia_dict, open(out_file, 'wb'))

    def count_valid_words(self, vocab, max_text_len):
        tokenizer = RegexpTokenizer()
        keys = self.keys()
        words = frozenset(list(vocab.words()))
        word_count = 0

        with click.progressbar(keys) as bar:
            for key in bar:
                c = 0
                for token in tokenizer.tokenize(self[key]['text']):
                    if token.text.lower() in words:
                        c += 1

                word_count += min(c, max_text_len)

        return word_count

def _process_file(file_name, in_dir):

    # Matches the hyperlink of the abstracts such as <http://dbpedia.org/resource/Anarchism/abstract#offset_0_1313>
    abs_matcher = re.compile(r'^http://dbpedia\.org/resource/(.*)/abstract#offset_(\d+)_(\d+)$')

    # Matches other dbpedia resource links such as <http://dbpedia.org/resource/Stateless_society> 
    # This also matches all strings matched by abs_matcher.
    dbp_matcher = re.compile(r'^http://dbpedia\.org/resource/(.*)$')

    click.echo('Processing %s' % file_name)

    g = rdflib.Graph()
    with gzip.GzipFile(os.path.join(in_dir, file_name)) as f:
        g.load(f, format='turtle')

    texts = {}
    mentions = defaultdict(dict)
    mention_titles = defaultdict(dict)

    for (s, p, o) in g:
        s = str(s)
        p = str(p)
        o = str(o)

        abs_match_obj = abs_matcher.match(s)
        
        title = urllib.parse.unquote(abs_match_obj.group(1))
        title = title.replace(u'_', u' ').lower()

        if p == u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString':
            texts[title] = o

        elif p == u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf':
            span = (int(abs_match_obj.group(2)), int(abs_match_obj.group(3)))
            mentions[title][s] = (o, span)

        elif p == u'http://www.w3.org/2005/11/its/rdf#taIdentRef':
            match_obj = dbp_matcher.match(o)
            if match_obj:
                link_title = urllib.parse.unquote(match_obj.group(1))
                link_title = link_title.replace(u'_', u' ').lower()
                mention_titles[title][s] = link_title

    ret = []

    for (title, text) in texts.items():
        links = []
        for (key, link_title) in mention_titles[title].items():
            (name, span) = mentions[title][key]
            links.append((name, link_title, span))

        ret.append((title, dict(title=title, text=text, links=links)))

    return ret