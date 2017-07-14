#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:58:48 2017

@author: moamin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

file_read = open('pairall_pearson_coefficients.txt', 'r')

pearson = []
p_value = []
count = 0

significant_pairs = []
same_word_pair = []
pair_score_map = {}
end_flag = True
min_p_value = 0.0005
min_pearson = -0.08
max_pearson = 0.14
max_data_count = 10000000

for line in file_read:
    if len(line) == 0:
        break
    assert(end_flag == True)
    
    columns = line.strip().split(" ")
    word_one = columns[0]
    pearson.append(float(columns[1]))
    p_value.append(float(columns[2]))
    pair_score_map[word_one + ',' + word_one] = [pearson[-1], p_value[-1]]
    
    #if (pearson[-1] > max_pearson or pearson[-1] < min_pearson) and p_value[-1] < min_p_value:
    if p_value[-1] < min_p_value:
        same_word_pair.append([word_one, word_one, pearson[-1], p_value[-1]])
        significant_pairs.append([word_one, word_one, pearson[-1], p_value[-1]])
        
    count += 1
    
    for line in file_read:
        if line.strip() == "END":
            end_flag = True
            break
        
        columns = line.strip().split(" ")
        word_two = columns[0]
        pearson.append(float(columns[1]))
        p_value.append(float(columns[2])) 
        pair_score_map[word_one + ',' + word_two] = [pearson[-1], p_value[-1]]
        
        #if (pearson[-1] > max_pearson or pearson[-1] < min_pearson) and p_value[-1] < min_p_value:
        if p_value[-1] < min_p_value:
            significant_pairs.append([word_one, word_two, pearson[-1], p_value[-1]])
        count += 1    
        end_flag = False
        if count % max_data_count == 0:
            print("Done! ", max_data_count)
            #break
    '''
    if count % max_data_count == 0:
        print("Done! ", max_data_count)
        break
    '''
    
print(len(pearson), len(p_value))

significant_pairs.sort(key=lambda _tuple_: _tuple_[-2])
file_write = open('pearson_pvalue_significant_pairs.txt', 'w') 
for i in range(0, len(significant_pairs)):
    print_string = str(significant_pairs[i])
    #print(print_string + '\n')
    file_write.write(print_string + '\n')
file_write.close()

same_word_pair.sort(key=lambda _tuple_: _tuple_[-2])
print("Same Word Pairs: ", len(same_word_pair))
file_write = open('pearson_pvalue_sameword_pairs.txt', 'w') 
for i in range(0, len(same_word_pair)):
    print_string = str(same_word_pair[i])
    #print(print_string + '\n')
    file_write.write(print_string + '\n')
file_write.close()

#pearson = significant_pairs[:][2]
#p_value = significant_pairs[:][3]
'''
plt.hist(pearson)
plt.xlabel('Pearsonr')
plt.ylabel('Frequency')
plt.savefig("pearsonr_correlation_histogram.png")
plt.show()
#plt.close()

#plt.figure(1, figsize=(20, 16))
plt.hist(p_value)
plt.xlabel('P_value')
plt.ylabel('Frequency')
plt.savefig("pearsonr_pvalue_histogram.png")
plt.show()
#plt.close()

#plt.hist2d(pearson, p_value, bins=40, norm=LogNorm())
plt.hist2d(pearson, p_value, bins=40, norm=LogNorm())
plt.colorbar()
plt.xlabel('Pearsonr')
plt.ylabel('P_value')
plt.savefig("pearsonr_pvalue_allpair.png")
plt.show()

pearson = [significant_pairs[i][2] for i in range(0, len(significant_pairs))]
p_value = [significant_pairs[i][3] for i in range(0, len(significant_pairs))]
plt.hist2d(pearson, p_value, bins=40, norm=LogNorm())
plt.colorbar()
plt.xlabel('Pearsonr')
plt.ylabel('P_value')
plt.savefig("pearsonr_pvalue_significant.png")
plt.show()
'''

pearson = [same_word_pair[i][2] for i in range(0, len(same_word_pair))]
p_value = [same_word_pair[i][3] for i in range(0, len(same_word_pair))]
print("Pearsonr, Pvalue: ", (len(pearson), len(p_value)))

print("Pearsonr Count = ", len(pearson))
plt.figure(1, figsize=(12, 9))
plt.hist(pearson, bins=50)#, normed=True)
plt.xlabel('Pearsonr')
plt.ylabel('Frequency')
plt.savefig("pearsonr_correlation_histogram.png")
plt.show()

print("P_value Count = ", len(p_value))
plt.figure(1, figsize=(12, 9))
plt.hist(p_value, bins=50)#, normed=True)
plt.xlabel('P_value')
plt.ylabel('Frequency')
plt.savefig("pearsonr_pvalue_histogram.png")
plt.show()

plt.figure(1, figsize=(12, 9))
plt.plot(pearson)
plt.xlabel('Words Order')
plt.ylabel('Pearsonr Coefficients')
plt.savefig("pearsonr_correlation_coefficients.png")
plt.show()

#print("Significantly Correlated 100 Words: ", same_word_pair[-100:])

'''
plt.hist2d(pearson, p_value, bins=40, norm=LogNorm())
plt.colorbar()
plt.xlabel('Pearsonr')
plt.ylabel('P_value')
plt.savefig("pearsonr_pvalue_hist2d_sameword.png")
plt.show()
'''