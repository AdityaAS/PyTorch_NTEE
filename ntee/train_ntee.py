# -*- coding: utf-8 -*-
# @Author: AdityaAS
# @Date:   2018-06-25 16:09:29
# @Last Modified by:   AdityaAS
# @Last Modified time: 2018-07-06 23:27:22

import click
import joblib
import numpy as np
from numpy.linalg import norm
from sklearn.cross_validation import train_test_split

import torch.optim as optim
import torch
from torch.autograd import Variable
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Embedding

from io import open
import os

from ntee.utils.sentence_detector import OpenNLPSentenceDetector
from ntee.utils.my_tokenizer import RegexpTokenizer
from models import NTEE
import pdb

from joblib import Parallel, delayed
import multiprocessing
import tables

def normalize(title):
	return title.replace('_', ' ').lower()

def pad_sequences(listoflists, max_len, pad_mode='pre', pad_value=0.0, dtype='int64'):
	padded_sequence = np.zeros(shape=(len(listoflists), max_len), dtype='int64')

	for index, l in enumerate(listoflists):
		if(len(l) < max_len):
			diff = max_len - len(l)
			if pad_mode == 'pre':
				padded_sequence[index, diff:] = l
			else:
				padded_sequence[index, :diff] = l
		else:
			if pad_mode == 'pre':
				padded_sequence[index, :] = l[0:max_len]
			else:
				padded_sequence[index, :] = l[-max_len:]

	return padded_sequence

def custom_concat(batches):
	
	new_word_batch = batches[0][0][0]
	new_entity_batch = batches[0][0][1]
	new_labels = batches[0][1]

	for i in range(1, len(batches)):
		np.concatenate(new_word_batch, batches[i][0][0])
		np.concatenate(new_entity_batch, batches[i][0][1])
		np.concatenate(new_labels, batches[i][1])

	return ([new_word_batch,
						new_entity_batch], new_labels)

import time
import GPUtil
def train_ntee(db, entity_db, vocab, word2vec, out_file, mode, text_len, dim_size,
		  negative, epoch, batch_size, word_static, entity_static, include_title,
		  optimizer, lr, dev_size, patience, num_links, random_seed, gpu):
	
	np.random.seed(random_seed)
	click.echo('Initializing weights...')
	print(batch_size)
	print(negative)
	print(dim_size)

	word_embedding = np.random.uniform(low=-0.05, high=0.05,
									   size=(vocab.word_size, dim_size))

	# Adding UNK to the vocabulary. UNK has embedding 0...
	word_embedding = np.vstack([np.zeros(dim_size), word_embedding])
	word_embedding = word_embedding.astype('float32')

	entity_embedding = np.random.uniform(low=-0.05, high=0.05,
										 size=(vocab.entity_size, dim_size))
	entity_embedding = entity_embedding.astype('float32')

	if word2vec:
		for word in vocab.words():
			try:
				vec = word2vec.get_word_vector(word)
			except KeyError:
				continue

			if vec is not None:
				word_embedding[vocab.get_word_index(word) + 1] = vec

		for entity in vocab.entities():
			try:
				vec = word2vec.get_entity_vector(entity)
			except KeyError:
				continue
			if vec is not None:
				entity_embedding[vocab.get_entity_index(entity)] = vec / norm(vec, 2)


	tokenizer = RegexpTokenizer()
	
	print(word_embedding.shape)
	print(entity_embedding.shape)
	
	if mode == 'sentence':
		sentence_detector = OpenNLPSentenceDetector()


	(train_keys, dev_keys) = train_test_split(list(db.keys()), test_size=dev_size,
										  random_state=random_seed)

	# import pdb; pdb.set_trace()
	# This is where the Training Code will start...

	# Weights:
		# Word Embeddings
		# Entity Embeddings
		# W (Text Layer)
		# b (Text Layer)
	# Input:
		# Training Batch

	model = NTEE(word_embedding, entity_embedding, 30, text_len, dim_size)
	model.cuda()
	print(model.parameters())

	optimizer  = optim.Adam(model.parameters(), lr=0.01)
	loss_function = nn.MSELoss()

	epochs = 2
	path = '/scratchd/home/adityaas/MyResearch/KGE/KG-Text/PyTorch_NTEE/ntee_data_precomputed/'

	word_file = tables.open_file(path + 'ntee_word_paragraph_batch.h5', mode='r')
	entity_file = tables.open_file(path + 'ntee_entity_paragraph_batch.h5', mode='r')
	label_file = tables.open_file(path + 'ntee_label_paragraph_batch.h5', mode='r')
	
	batchno = 1

	totalbatches = word_file.root.data.shape[0] / batch_size 

	start = 0
	starttime = time.time()

	while start < word_file.root.data.shape[0]:
		batchstart = time.time()

		mini_word_batch = word_file.root.data[start:start+batch_size, :]
		mini_entity_batch = entity_file.root.data[start:start+batch_size, :]
		mini_label_batch = label_file.root.data[start:start+batch_size, :]

		optimizer.zero_grad()

		words = torch.from_numpy(mini_word_batch.astype(int)).cuda()
		entities = torch.from_numpy(mini_entity_batch.astype(int)).cuda()
		labels = torch.from_numpy(mini_label_batch.astype(int)).cuda()
		probs = model(words, entities, labels)

		loss = loss_function(probs, Variable(labels.float()))
		
		loss.backward()
		
		optimizer.step()
		
		start = start + batch_size
		if batchno % 10000:
			print("Time for batch: " + str(batchno) + " / " + str(int(totalbatches)) + " is: " + str(time.time()-batchstart))
		
		print(loss.data)
		batchno = batchno + 1

	print("Time for one Epoch: ", time.time() - starttime)
	import pdb; pdb.set_trace()
	torch.save(model.state_dict(), out_file)