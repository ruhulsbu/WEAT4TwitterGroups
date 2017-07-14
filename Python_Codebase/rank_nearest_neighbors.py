import os, sys, time, gzip, math
import numpy as np,csv, json, re
from gensim.models import Word2Vec

import matplotlib
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

for word in keywords:
    print("Word: ", word)
    #first_neighbors_set = set(first_word2vec_model.wv.most_similar(word, topn=10))
    #print(first_neighbors_set)
    #continue
    
    similar_word_found = False
    for i in range(1, len(top_n_words)):    
        first_neighbors_list = first_word2vec_model.wv.most_similar(word, topn=i)
        first_neighbors_set = set([word[0] for word in first_neighbors_list])
        try:
            second_neighbors_list = second_word2vec_model.wv.most_similar(word, topn=i)
            second_neighbors_set = set([word[0] for word in second_neighbors_list])
        except:
            continue
        
        #print(first_neighbors_set)
        #print(second_neighbors_set)
        
        
        intersection_set = first_neighbors_set & second_neighbors_set
        #print("Intersection: %d: %d " %(i, len(intersection_set)))
        if len(intersection_set) > 0:
            top_n_words[i] += len(intersection_set)
            similar_word_found = True
                    
    if similar_word_found == True:
        model_intersection_words.append(word)
    if len(model_intersection_words) >= max_words:
        break

percentage = [0 for i in range(0, n_words+1)]
logscale = [0 for i in range(0, n_words+1)]
for i in range(1, len(top_n_words)):
    percentage[i] = 100.0 * top_n_words[i] / (i * len(model_intersection_words))
    logscale[i] = math.log10(top_n_words[i])
    
print("Total Words: ", len(model_intersection_words))
print("--------------------------------------------")
print(top_n_words)

#plt.figure(1, figsize=(20, 16))
plt.plot(top_n_words)
plt.xlabel('Nearest Neighbors')
plt.ylabel('Frequency of Words in Agreement')
plt.savefig("agreement_top_n_words.png")
plt.close()
#plt.show()

#plt.figure(1, figsize=(20, 16))
plt.plot(percentage)
plt.xlabel('Nearest Neighbors')
plt.ylabel('Frequency of Words in Agreement')
plt.savefig("agreement_percentage")
plt.close()
#plt.show()

#plt.figure(1, figsize=(20, 16))
plt.plot(logscale)
plt.xlabel('Nearest Neighbors')
plt.ylabel('Frequency of Words in Agreement')
plt.savefig("agreement_logscale")
plt.close()
#plt.show()
