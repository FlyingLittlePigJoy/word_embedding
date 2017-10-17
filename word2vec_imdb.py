#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec, LineSentence
from gensim import utils

import numpy as np
import random
import logging
from sklearn.linear_model import LogisticRegression
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s:%(message)s')

class LineSentences:
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('prefixes collapse.')
    
    def get_array(self):
        self.sentences = []
        for key, value in self.sources.items():
            self.sentences.extend(LineSentence(key))
            
        return self.sentences
    
    def perm_sentences(self):
        random.shuffle(self.sentences)
        return self.sentences
    
class Model:
    def __init__(self, linesentence, n_epoches=10, n_dims=100):
        self.linesentence = linesentence
        self.n_epoches = n_epoches
        self.model = Word2Vec(min_count=1, size=n_dims, window=8, negative=5, sample=1e-5, workers=8)
        
    def _train_model(self):
        self.model.build_vocab(self.linesentence.get_array())
        for epoch in range(self.n_epoches):
            logger.info('epoch' + str(epoch + 1))
            self.model.train(self.perm_sentences(), 
                             total_examples=self.model.corpus_count, 
                             epochs=self.model.iter)
        
    def _save_model(self, path):
        self.model.save(path)
        
    def build(self, path):
        self._train_model()
        self._save_model(path)
        
class Data(object):
    def __init__(self, n_dims, wv_model_path):
        self.n_dims = n_dims
        self.wv_model = Word2Vec.load(wv_model_path)
        self.vocab = list(self.wv_model.wv.vocab.keys())
        self.inv_vocab = {word: ix for ix, word in enumerate(self.vocab)}
        
    def _get_embeddings(self):
        if not hasattr(self, 'embeddings'):
            self.embeddings = np.array([self.wv_model.wv[word] for word in self.vocab])
        return self.embeddings

    def _read_data(self, fpath, sign, method='mean'):
        data, label = [], []
        embeddings = self._get_embeddings()
        with utils.smart_open(fpath) as fin:
            for l_no, line in enumerate(fin):
                words = line.strip().split()
                words = [word.decode('utf8') for word in words]
                ixs = [self.inv_vocab[word] for word in words if word in self.vocab]
                vecs = embeddings[ixs]
                if method == 'mean':
                    vec = np.mean(vecs, axis=0)
                data.append(vec)
                label.append(sign)
        return data, label
    
    def _get_neg_pos(self, fpaths, savepath_data, savepath_label, method):
        t_data, t_label = [], []
        if os.path.exists(savepath_data) and os.path.exists(savepath_label):
            logger.info('load data and label from ' + str(savepath_data) + ',' + str(savepath_label))
            t_data, t_label = self._read2data_label(savepath_data, savepath_label)
            return t_data, t_label
        
        for ti in range(len(fpaths)):
            logger.info('get data from ' + fpaths[ti])
            data, label = self._read_data(fpaths[ti], ti, method)
            t_data.extend(data)
            t_label.extend(label)
        
        t_data, t_label = np.array(t_data), np.array(t_label)
        self._save2file(t_data, t_label, savepath_data, savepath_label)
        return t_data, t_label
    
    def get_train_test(self, savepath_trdata, savepath_trlabel, savepath_tedata, savepath_telabel, method='mean'):
        train_fpaths, test_fpaths = ['train-neg.txt', 'train-pos.txt'], ['test-neg.txt', 'test-pos.txt']
        train_data, train_label = self._get_neg_pos(train_fpaths, savepath_trdata, savepath_trlabel, method)
        test_data, test_label = self._get_neg_pos(test_fpaths, savepath_tedata, savepath_telabel, method)
        
        return train_data, train_label, test_data, test_label
    
    def _save2file(self, data, label, savepath_data, savepath_label):
        strdata = '\n'.join([' '.join([str(num) for num in line]) for line in data])
        strlabel = '\n'.join([str(num) for num in label])
        
        with open(savepath_data, 'wb') as fin:
            fin.write(strdata.encode('utf8'))
        with open(savepath_label, 'wb') as fin:
            fin.write(strlabel.encode('utf8'))
            
    def _read2data_label(self, savepath_data, savepath_label):
        data, label = [], []
        with open(savepath_data, 'rb') as fr:
            strdata = fr.readlines()
        with open(savepath_label, 'rb') as fr:
            strlabel = fr.readlines()
        
        data = np.array([[float(num) for num in line.strip().split()] for line in strdata])
        label = np.array([[int(float(num)) for num in line.strip().split()] for line in strlabel])
        label = label.reshape(-1, )
        return data, label
#==============================================================================
# main
#==============================================================================
if __name__ == '__main__':
    sources = {'train-pos.txt': 'TRAIN_POS', 'train-neg.txt': 'TRAIN_NEG', 
               'test-pos.txt': 'TEST_POS', 'test-neg.txt': 'TEST_NEG', 'train-unsup.txt': 'TRAIN_UNS'}
    n_epoches, n_dims, model_savepath= 1, 100, './my_imdb.w2v'
    
    savepath_trdata, savepath_trlabel, savepath_tedata, savepath_telabel = './tr.data', './tr.label', './te.data', './te.label'
    
    my_linesentences = LineSentences(sources)
    model_wv = Model(my_linesentences, n_epoches, n_dims)
    model_wv.build(model_savepath)
    
    data = Data(n_dims, model_savepath)
    train_data, train_label, test_data, test_label = data.get_train_test(savepath_trdata, savepath_trlabel, savepath_tedata, savepath_telabel)
        
    clf = LogisticRegression()
    clf.fit(train_data, train_label)
    print('classification accuracy is ', clf.score(test_data, test_label))


    
