#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:55:28 2017

@author: moamin
"""

import os, sys, time, gzip, math
import numpy as np,csv, json, re
from gensim.models import Word2Vec
import matplotlib, string
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

first_word2vec_model = Word2Vec.load('updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
second_word2vec_model = Word2Vec.load('updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

first_keywords = set(key for key in first_word2vec_model.wv.vocab.keys())
second_keywords = set(key for key in second_word2vec_model.wv.vocab.keys())

common_keywords = sorted(list(first_keywords & second_keywords))
common_worddict = {w:ind for ind, w in enumerate(common_keywords)}

first_n_words = len(first_keywords)
second_n_words = len(second_keywords)
#print("Total Common Keywords, First Set, Second Set, Dict: ", 
#	len(common_keywords), first_n_words, second_n_words, len(common_worddict))
#exit()
exclude = set(string.punctuation)

first_rank_dict = {}
second_rank_dict = {}


for word in common_keywords:
    first_neighbors_list = first_word2vec_model.wv.most_similar(word, topn=first_n_words)
    first_wordlist = [w[0] for w in first_neighbors_list if w[0] in common_worddict]
    first_rank_dict[word] = [common_worddict[w] for w in first_wordlist]

    second_neighbors_list = second_word2vec_model.wv.most_similar(word, topn=second_n_words)
    second_wordlist = [w[0] for w in second_neighbors_list if w[0] in common_worddict] 
    second_rank_dict[word] = [common_worddict[w] for w in second_wordlist]


for ind in range(0, len(common_keywords)):
    word_one = common_keywords[ind]
    print_word = ''.join(ch for ch in word_one if ch not in exclude)
    if len(print_word) < 1:
        #print("Excluding: ", word_one)
        continue
        print(word_one)
    first_rank = first_rank_dict[word_one]

    for knd in range(ind, len(common_keywords)):
        word_two = common_keywords[knd]
        print_word = ''.join(ch for ch in word_two if ch not in exclude)
        if len(print_word) < 1:
            #print("Excluding: ", word_two)
            continue
        second_rank = second_rank_dict[word_two]

        #print("First/Second = ", (len(first_rank), len(second_rank)))  
        coefficient, pvalue = pearsonr(first_rank, second_rank)
        print(word_two, coefficient, pvalue)
        #break

    print("END")
    #break
    
