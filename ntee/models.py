 # -*- coding: utf-8 -*-
# @Author: AdityaAS
# @Date:   2018-06-05 11:42:16
# @Last Modified by:   adityaas
# @Last Modified time: 2018-07-09 12:17:02

import torch
from torch.autograd import Variable
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from io import open
import os
from torch.nn import Embedding
import pdb

class NTEE(nn.Module):
    def __init__(self, word_embedding, entity_embedding, negative_size, text_len, dim_size, W=None, b=None):
        super(NTEE, self).__init__()

        self.negative_size = negative_size
        self.text_len = text_len

        self.word_size = word_embedding.shape[0]
        self.entity_size = entity_embedding.shape[0]

        # Input to these embedding layers are long tensor of arbitrary (2000) shape
        self.word_embedding = Embedding(word_embedding.shape[0], word_embedding.shape[1], _weight=torch.from_numpy(word_embedding))
        self.entity_embedding = Embedding(entity_embedding.shape[0], entity_embedding.shape[1], _weight=torch.from_numpy(entity_embedding))
        self.W = nn.Parameter(torch.from_numpy(np.identity(dim_size)).float().cuda())
        self.b = nn.Parameter(torch.from_numpy(np.random.uniform(-0.05, 0.05, dim_size)).float().cuda())

    def forward(self, text_input, entity_input, labels):
        
        def text_transform(i, embedding, text_input):
            mask = torch.ne(text_input, 0.0).float()
            vec = torch.matmul(mask, embedding)
            vec = vec / torch.norm(vec).float()
            return torch.matmul(vec, self.W) + self.b
            # s
        text_transformed_input = torch.stack([text_transform(torch.tensor(i), self.word_embedding(text_input[i]), text_input[i]) for (i, x) in enumerate(text_input)])
        similarity = torch.stack([torch.matmul(text_transformed_input[i], torch.transpose(self.entity_embedding(entity_input)[i], 0, 1)) for i in range(text_input.shape[0])])
        predictions = F.softmax(similarity)

        return predictions
            
        # def text_transform(i, embedding, text_input):
        #     mask = torch.ne(text_input, 0.0).float()
        #     vec = torch.matmul(mask, embedding)
        #     vec = vec / torch.norm(vec).float()
        #     return torch.matmul(vec, self.W) + self.b

        # import pdb; pdb.set_trace()
        # text_transformed_input = torch.stack([text_transform(torch.tensor(i), self.word_embedding(text_input[i]), text_input[i]) for (i, x) in enumerate(text_input)])
        # similarity = torch.stack([torch.matmul(text_transformed_input[i], torch.transpose(self.entity_embedding(entity_input)[i], 0, 1)) for i in range(text_input.shape[0])])
        # predictions = F.softmax(similarity)

