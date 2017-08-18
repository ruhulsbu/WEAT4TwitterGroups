#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:55:28 2017

@author: moamin
"""

import os, sys, time, gzip, math
import numpy as np,csv, json, re
#from gensim.models import Word2Vec
import matplotlib.pyplot as plt
#from sets import Set

from representations.sequentialembedding import SequentialEmbedding

"""
Example showing how to load a series of historical embeddings and compute similarities over time.
Warning that loading all the embeddings into main memory can take a lot of RAM
"""

def histword_similarity(target_word, attribute_word):
    fiction_embeddings = SequentialEmbedding.load("embeddings/eng-fiction-all_sgns", range(1900, 2000, 10))
    time_sims = fiction_embeddings.get_time_sims(target_word, attribute_word)
    #print "Similarity between gay and lesbian drastically increases from 1950s to the 1990s:"
    for year, sim in time_sims.iteritems():
        print("{year:d}, cosine similarity={sim:0.2f}".format(year=year,sim=sim))


def cross_target_attribute(target, target_words, attribute, attribute_words):
    for i in range(0, len(target_words)):
        for k in range(0, len(attribute_words)):
            wt = target_words[i][:-1]
            at = attribute_words[k][:-1]
            '''
            try:
                lib_cosine = lib_word2vec_model.similarity(wt, at)
                con_cosine = con_word2vec_model.similarity(wt, at)
            except:
                continue
            #print(wt, target, at, attribute, lib_cosine, con_cosine)
            print_txt = wt + ',' + target + ',' + at + ',' + attribute + ',' + str(lib_cosine) + ',' + str(con_cosine)
            print(print_txt)
            '''
            try:
                print_txt = wt + ',' + target + ',' + at + ',' + attribute
                print(print_txt) 
                histword_similarity(wt, at)
            except:
                print("Exception: Not Present")
                continue

'''
lib_word2vec_model = Word2Vec.load('../../updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
con_word2vec_model = Word2Vec.load('../../updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])
#cosine_similarity = model.similarity(word1,word2)
print('Wt, Ct, Wa, Ca, Liberals, Conservatives')
'''

input_file = './weat_file.txt'
file_read = open(input_file, 'r')

count = 0
for line in file_read:
    if len(line.strip()) == 0:
        break
    #print(line)
    count += 1
    if count == 1:
        continue
    if count == 2:
        words = line.split()
        target_one = words[0][:-1]
        target_one_words = words[1:]
        
    if count == 3:
        words = line.split()
        target_two = words[0][:-1]
        target_two_words = words[1:]
        
    if count == 4:
        words = line.split()
        attribute_one = words[0][:-1]
        attribute_one_words = words[1:]
        
    if count == 5:
        words = line.split()
        attribute_two = words[0][:-1]
        attribute_two_words = words[1:]
        
        count = 0
        
    if count == 0:
        #print(target_one, attribute_one, "(Format: Wt, Ct, Wa, Ca, Liberals, Conservatives)")
        #print('===============================================================================')
        cross_target_attribute(target_one, target_one_words, attribute_one, attribute_one_words)
        
        #print(target_two, attribute_two, "(Format: Wt, Ct, Wa, Ca, Liberals, Conservatives)")
        #print('===============================================================================')
        cross_target_attribute(target_two, target_two_words, attribute_two, attribute_two_words)
                
        #break
