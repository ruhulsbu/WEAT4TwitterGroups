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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sets import Set

first_word2vec_model = Word2Vec.load('updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
second_word2vec_model = Word2Vec.load('updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

keywords = [key for key in first_word2vec_model.wv.vocab.keys()]
total_percentage_similar_list = []
print("Total Keywords: ", len(keywords))

n_words = 100
max_words = int(sys.argv[1])
top_n_words = [0 for i in range(0, n_words+1)]
model_intersection_words = []

file_first = open("disagreement_in_words_liberals.csv", "w")
file_second = open("disagreement_in_words_conservatives.csv", "w")

neighbor_first = open("neighbor_words_for_liberals.csv", "w")
neighbor_second = open("neighbor_words_for_conservatives.csv", "w")

exclude = set(string.punctuation)

for word in keywords:
    print("Word: ", word)
    print_word = ''.join(ch for ch in word if ch not in exclude)
    if len(print_word) < 1:
        continue
    #first_neighbors_set = set(first_word2vec_model.wv.most_similar(word, topn=10))
    #print(first_neighbors_set)
    #continue
    
    similar_word_found = False
    for i in range(1, len(top_n_words)):    
        first_neighbors_list = first_word2vec_model.wv.most_similar(word, topn=i)
        first_neighbors_set = set([w[0] for w in first_neighbors_list])
        try:
            second_neighbors_list = second_word2vec_model.wv.most_similar(word, topn=i)
            second_neighbors_set = set([w[0] for w in second_neighbors_list])
        except:
            continue
        
        #print(first_neighbors_set)
        #print(second_neighbors_set)
        
        
        intersection_set = first_neighbors_set & second_neighbors_set
        #print("Intersection: %d: %d " %(i, len(intersection_set)))
        if len(intersection_set) > 0:
            top_n_words[i] += 1
            similar_word_found = True

            first_neighbors_list = first_word2vec_model.wv.most_similar(word, topn=100)
            first_neighbors_set = set([w[0] for w in first_neighbors_list])

            second_neighbors_list = second_word2vec_model.wv.most_similar(word, topn=100)
            second_neighbors_set = set([w[0] for w in second_neighbors_list])

            intersection_set = first_neighbors_set & second_neighbors_set
            print_word += "," + str(len(intersection_set)) + "," + str(i)

            str_first = print_word + "," + ','.join(str(n) for n in first_word2vec_model[word]) + "\n"
            file_first.write(str_first)

            str_second = print_word + "," + ','.join(str(n) for n in second_word2vec_model[word]) + "\n"
            file_second.write(str_second)

            neighbor_first.write(print_word + "," + ','.join(w[0] for w in first_neighbors_list) + "\n")
            neighbor_second.write(print_word + "," + ','.join(w[0] for w in second_neighbors_list) + "\n")

            assert(len(str_first.split(",")) == 103)
            assert(len(str_second.split(",")) == 103)

            break
                    
    if similar_word_found == True:
        model_intersection_words.append(word)
    if len(model_intersection_words) >= max_words:
        break

percentage = [0 for i in range(0, n_words+1)]
logscale = [0 for i in range(0, n_words+1)]
for i in range(1, len(top_n_words)):
    percentage[i] = 100.0 * top_n_words[i] / len(model_intersection_words)
    logscale[i] = math.log10(top_n_words[i]+1)
    
print("Total Words: ", len(model_intersection_words))
print("--------------------------------------------")
print(top_n_words)
'''
#plt.figure(1, figsize=(20, 16))
plt.plot(top_n_words)
plt.xlim(1, len(top_n_words))
plt.xlabel('Nearest Neighbors')
plt.ylabel('Rank of First Agreement')
plt.savefig("rank_agreement_top_n_words.png")
#plt.show()
plt.close()

#plt.figure(1, figsize=(20, 16))
plt.plot(percentage)
plt.xlim(1, len(top_n_words))
plt.xlabel('Nearest Neighbors')
plt.ylabel('Rank of First Agreement')
plt.savefig("rank_agreement_percentage")
#plt.show()
plt.close()

#plt.figure(1, figsize=(20, 16))
plt.plot(logscale)
plt.xlim(1, len(top_n_words))
plt.xlabel('Nearest Neighbors')
plt.ylabel('Rank of First Agreement')
plt.savefig("rank_agreement_logscale")
#plt.show()
plt.close()
'''
