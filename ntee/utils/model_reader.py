# -*- coding: utf-8 -*-

import joblib
import torch
import numpy as np
from ntee.utils.my_tokenizer import RegexpTokenizer
from ntee.utils.vocab_joint import JointVocab
from ntee.utils.vocab import Vocab

class ModelReader(object):
    def __init__(self, model_file):
        model = joblib.load(model_file, mmap_mode='r')
        self._word_embedding = model['word_embedding']
        self._entity_embedding = model['entity_embedding']
        self._W = model.get('W')
        self._b = model.get('b')
        self._vocab = model.get('vocab')

        self._tokenizer = RegexpTokenizer()

    @property
    def vocab(self):
        return self._vocab

    @property
    def word_embedding(self):
        return self._word_embedding

    @property
    def entity_embedding(self):
        return self._entity_embedding

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    def get_word_vector(self, word, default=None):
        index = self._vocab.get_word_index(word)
        if index:
            return self.word_embedding[index]
        else:
            return default

    def get_entity_vector(self, title, default=None):
        index = self._vocab.get_entity_index(title)
        if index:
            return self.entity_embedding[index]
        else:
            return default

    def get_text_vector(self, text):
        vectors = [self.get_word_vector(t.text.lower())
                   for t in self._tokenizer.tokenize(text)]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return None

        ret = np.mean(vectors, axis=0)
        ret = np.dot(ret, self._W)
        ret += self._b

        ret /= np.linalg.norm(ret, 2)

        return ret

class PyTorchModelReader(object):
    def __init__(self, model_file, vocab_file, modelname, cpuflag=False):
        model = torch.load(model_file)
        
        if cpuflag: #CPU Tensors
            self._word_embedding = model['word_embedding.weight'].cpu()
            self._entity_embedding = model['entity_embedding.weight'].cpu()

            if modelname.lower() == 'ntee':
                self._vocab = Vocab.load(vocab_file)
            else:
                self._relation_embedding = model['relation_embedding.weight'].cpu()
                self._vocab = JointVocab.load(vocab_file)
        
            self._W = model.get('W').cpu()
            self._b = model.get('b').cpu()

        else: #GPU Tensors
            self._word_embedding = model['word_embedding.weight']
            self._entity_embedding = model['entity_embedding.weight']

            if modelname.lower() == 'ntee':
                self._vocab = Vocab.load(vocab_file)
            else:
                self._relation_embedding = model['relation_embedding.weight']
                self._vocab = JointVocab.load(vocab_file)
        
            self._W = model.get('W')
            self._b = model.get('b')

        self._tokenizer = RegexpTokenizer()

    @property
    def vocab(self):
        return self._vocab

    @property
    def word_embedding(self):
        return self._word_embedding

    @property
    def entity_embedding(self):
        return self._entity_embedding

    @property
    def relation_embedding(self):
        return self._relation_embedding
    
    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    def get_word_vector(self, word, default=None):
        index = self._vocab.get_word_index(word)
        if index:
            return self.word_embedding[index]
        else:
            return default

    def get_entity_vector(self, title, default=None):
        index = self._vocab.get_entity_index(title)
        if index:
            return self.entity_embedding[index]
        else:
            return default

    def get_text_vector(self, text):

        vectors = [self.get_word_vector(t.text.lower())
                   for t in self._tokenizer.tokenize(text)]

        vectors = [v for v in vectors if v is not None]
        
        # vectors_numpy = [v.cpu().numpy() for v in vectors if v is not None]
        # ret_numpy = np.mean(vectors_numpy, axis=0)
        # ret_numpy = np.dot(ret_numpy, self._W.cpu().numpy())
        # ret_numpy += self._b.cpu().numpy()
        # ret_numpy /= np.linalg.norm(ret_numpy, 2)
        # return ret_numpy
        
        if not vectors:
            return None

        ret = torch.zeros(vectors[0].shape)
        
        for v in vectors:
            ret += v
        ret = ret / len(vectors)

        ret = torch.matmul(ret, self._W)
        ret += self._b

        ret /= torch.norm(ret, 2)
        return ret
