#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:55:28 2017

@author: moamin
"""

import os, sys, time, gzip, math
import numpy as np,csv, json, re
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
#from sets import Set

def cross_target_attribute(target, target_words, attribute, attribute_words):
    for i in range(0, len(target_words)):
        for k in range(0, len(attribute_words)):
            wt = target_words[i][:-1]
            at = attribute_words[k][:-1]
            try:
                lib_cosine = lib_word2vec_model.similarity(wt, at)
                con_cosine = con_word2vec_model.similarity(wt, at)
            except:
                continue
            #print(wt, target, at, attribute, lib_cosine, con_cosine)
            print_txt = wt + ',' + target + ',' + at + ',' + attribute + ',' + str(lib_cosine) + ',' + str(con_cosine)
            print(print_txt)

lib_word2vec_model = Word2Vec.load('../updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
con_word2vec_model = Word2Vec.load('../updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])
#cosine_similarity = model.similarity(word1,word2)

print('Wt, Ct, Wa, Ca, Liberals, Conservatives')
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
        target_two = words[0]
        target_two_words = words[1:]
        
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
