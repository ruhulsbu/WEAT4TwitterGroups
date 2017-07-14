#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:05:40 2017

@author: moamin
"""

# gensim modules
import gensim
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from random import shuffle
from sklearn.linear_model import LogisticRegression
import logging, os, re, sys, numpy, gzip
'''
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
 '''
#https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/

def get_doc_list(folder_name):
    doc_list = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('.gz')]
    for file in file_list:
        #st = gzip.open(file,'r').read()
        #doc_list.append(st)
        doc_list.append(file)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list
 
def get_doc(folder_name, doc_type):
 
    doc_list = get_doc_list(folder_name)
    tagged_doc = []
    original_doc = []
    doc_no = 0
    
    for index, doc in enumerate(doc_list):
        # for tagged doc
        with utils.smart_open(doc) as fin:
            for item_no, line in enumerate(fin):
                #yield TaggedDocument(utils.to_unicode(line).split(), [doc_type + '_%s' % item_no])
                td = TaggedDocument(utils.to_unicode(line).split(), [doc_type + '_%s' % doc_no])
                tagged_doc.append(td)
                
                doc_no += 1
                original_doc.append(line)
                assert(len(original_doc) == doc_no)
 
    print("Total Docs: ", len(tagged_doc), len(original_doc), doc_no)
    return tagged_doc, original_doc


def create_model(path, doc_type):
    tagged_doc, doc_list = get_doc(path, doc_type)
    print ('Data Loading finished: ', doc_type) 

    model = Doc2Vec(min_count=10, window=10, size=100, dbow_words=1)
    model.build_vocab(tagged_doc)
    
    for epoch in range(20):
        shuffle(tagged_doc)
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(tagged_doc, total_examples=model.corpus_count, epochs=5)
        model.alpha -= 0.0002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        
    model.save('./' + doc_type + '_docvec.d2v')
    return model, tagged_doc, doc_list
    
#liberal_model = Doc2Vec.load('./' + doc_type + '_docvec.d2v')
#conservative_model = Doc2Vec.load('./' + doc_type + '_docvec.d2v')

liberal_model, liberal_tagged_list, liberal_doc_list = create_model(\
        '/gpfs/scratch/moamin/Jason_Tweets/updated_liberals_wordvec/dataset_liberals',\
        'liberal')
conservative_model, conservative_tagged_list, conservative_doc_list =  create_model(\
        '/gpfs/scratch/moamin/Jason_Tweets/updated_conservatives_wordvec/dataset_conservatives',\
        'conservative')

print("Model Created and Saved...")

#similars = model.docvecs.most_similar(positive=[model.infer_vector(doc_words)])
#https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/
#https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1

#https://github.com/Akirato/Twitter-Sentiment-Analysis-Tool/blob/master/sentiment_analyser.py
#https://github.com/yazquez/poc-machine-learning/blob/master/Gensim.py
#http://linanqiu.github.io/2015/10/07/word2vec-sentiment/


'''
# build the model

model_liberal = Doc2Vec(min_count=10, window=10, size=100, dbow_words=1)
model_liberal.build_vocab(liberal_docs)

# start training
for epoch in range(10):
    shuffle(liberal_docs)
    if epoch % 2 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(liberal_docs, total_examples=model.corpus_count, epochs=1)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.save('./imdb.d2v')

model = Doc2Vec.load('./imdb.d2v')
# shows the similar words
print (model.most_similar('#p2b'))
# shows the learnt embedding
print (model['#p2b'])

'''
'''
# shows the similar docs with id = 2
print (model.docvecs.most_similar(str(2)))

model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)

log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

#https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/


log.info('Sentiment')
train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

print(train_labels)

test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(test_arrays, test_labels))
'''