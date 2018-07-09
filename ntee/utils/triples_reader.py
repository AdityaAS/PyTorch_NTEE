import bz2
import logging
import re
import json
from gensim.corpora import wikicorpus
import pickle
logger = logging.getLogger(__name__)

class TripleReader(object):
    def __init__(self, dump_file, formatt='tab'):
        self._dump_file = dump_file
        self._formatt = formatt

    @property
    def dump_file(self):
        return self._dump_file

    def _normalize_title(self, title):
        return title.lower()

    def __iter__(self, formatt='tab'):
        with open(self._dump_file) as f:
            c = 0
            if self._formatt == 'json':
                for line in f:
                    c += 1
                    line_dict = json.loads(line)

                    yield (str(self._normalize_title(line_dict['h_name'])), str(line_dict['r']), str(self._normalize_title(line_dict['t_name'])))

                    if c % 10000 == 0:
                        logger.info('Processed: %d', c)
            
            elif self._formatt == 'tab':
                for line in f:
                    c += 1

                    h, t, r = line.strip().split('\t')
                    yield (str(h), str(r), str(t))

                    if c % 10000 == 0:
                        logger.info('Processed: %d', c)

            else:
                print("Format not recognized")
                quit()