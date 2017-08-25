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
import matplotlib, random, itertools
import matplotlib.pyplot as plt


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
    return time_sims

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

fig = plt.figure(figsize=(12, 12))
colorno = 0
number = 20
cmap = plt.get_cmap(sys.argv[1])
colors = [cmap(i) for i in np.linspace(0, 1, number)]

for k in range(2, len(sys.argv)):
    input_file = sys.argv[k]
    file_read = open(input_file, 'r')
    for line in file_read:
        word = line.strip()
        print('woman vs ' + word)
        time_sims_woman = histword_similarity(word, 'woman')

        print('man vs ' + word)
        time_sims_man = histword_similarity(word, 'man')

        print(len(time_sims_woman), len(time_sims_man))
    
        #print(time_sims_woman)
        x = []
        y = []
        for year in time_sims_woman:
            sim_w = time_sims_woman[year]
            dit_w = 1 - sim_w
            sim_m = time_sims_man[year]
            dit_m = 1 - sim_m
            print(year, sim_w - sim_m, dit_w - dit_m)
            x.append(year)
            y.append(dit_w - dit_m)

        plt.plot(y, 'o-', label=word, color=colors[colorno])
        colorno = (colorno + 1) % number
        #break    

plt.xticks([i for i in range(0, 10)], [str(y) for y in range(1900, 2000, 10)], rotation='vertical')
plt.plot([0 for i in range(0, 10)], 'k--', label='No Bias')
plt.xlabel('Year')
plt.ylabel('(Woman -ve) - Bias - (Man + )')
#plt.legend(loc='best', mode='expand')
#plt.legend(loc = 1, fontsize=8) #bbox_to_anchor=(0.90, 0.6))#, borderaxespad=0.05)
plt.legend(loc=7, fontsize=8) #, bbox_to_anchor=(1.0, 1.0))
plt.xlim(0, 12)
plt.ylim(-0.25, 0.25)
#plt.savefig("gender_bias_" + word + ".png")
plt.show()
exit()
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
