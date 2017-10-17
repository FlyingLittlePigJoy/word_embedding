#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim import utils

import numpy as np
import random

from sklearn.linear_model import LogisticRegression

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('The prefixes colapsed.')
        
    def get_array(self):
        self.sentences = []
        for key, value in self.sources.items():
            print('File: ' + key)
            with utils.smart_open(key) as fin:
                for item_n, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [value + '_' + str(item_n)]))
                    
        return self.sentences
                    
    def sentence_perm(self):
        random.shuffle(self.sentences)
        return self.sentences
    
class Model(object):
    def __init__(self, labeledlinesentence, n_epoches=10, n_dims=100):
        self.labeledlinesentence = labeledlinesentence
        self.model = Doc2Vec(min_count=1, window=8, size=n_dims, sample=1e-5, negative=5, workers=8)
        self.n_epoches = n_epoches
        
    def train_vec(self):
        self.model.build_vocab(self.labeledlinesentence.get_array())
        for epoch in range(self.n_epoches):
            print('epoch %s' % (epoch+1))
            self.model.train(self.labeledlinesentence.sentence_perm(), 
                             total_examples=self.model.corpus_count, 
                             epochs=self.model.iter)
            
    def save_model(self, save_path):
        self.model.save(save_path)
        
    def build(self, save_path):
        self.train_vec()
        self.save_model(save_path)

class Data(object):
    def __init__(self, n_dims, dv_model_path):
        self.n_dims = n_dims
        self.dv_model = Doc2Vec.load(dv_model_path)
        
    def _read_data(self, prefix, sign, line_no=12500):
        data = np.zeros((line_no, self.n_dims))
        label = np.ones(line_no) * sign
        for ln in range(line_no):
            data[ln] = self.dv_model.docvecs[prefix + '_' + str(ln)]
        return data, label
    
    def _get_neg_pos(self, prefixes):
        t_data, t_label = [], []
        for ti in range(len(prefixes)):
            data, label = self._read_data(prefixes[ti], ti)
            t_data.extend(data)
            t_label.extend(label)
        return t_data, t_label
    
    def get_train_test(self):
        train_prefixes, test_prefixes = ['TRAIN_NEG', 'TRAIN_POS'], ['TEST_NEG', 'TEST_POS']
        train_data, train_label = self._get_neg_pos(train_prefixes)
        test_data, test_label = self._get_neg_pos(test_prefixes)
        
        return train_data, train_label, test_data, test_label
#==============================================================================
# main
#==============================================================================
if __name__ == '__main__':
    sources = {'train-pos.txt': 'TRAIN_POS', 'train-neg.txt': 'TRAIN_NEG', 'test-pos.txt': 'TEST_POS', 'test-neg.txt': 'TEST_NEG', 'train-unsup.txt': 'TRAIN_UNS'}
    n_epoches, n_dims,  model_savepath= 50, 100, './my_imdb.d2v'
    
    my_labeledlinesentence = LabeledLineSentence(sources)
    model_dv = Model(my_labeledlinesentence, n_epoches, n_dims)
    model_dv.build(model_savepath)
    
    data = Data(n_dims, model_savepath)
    train_data, train_label, test_data, test_label = data.get_train_test()
    
    clf = LogisticRegression()
    clf.fit(train_data, train_label)
    print('classification accuracy is ', clf.score(test_data, test_label))
