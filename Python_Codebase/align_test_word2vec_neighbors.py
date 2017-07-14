def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
            (With help from William. Thank you!)

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(
        base_embed, other_embed, words=words)
    #return in_base_embed, in_other_embed
    #print("SVD Calculation and Transformation:")
    
    # get the embedding matrices
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm
    
    '''
    #base_vecs = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    #other_vecs = np.array([[11, 12, 13],[14, 15, 16],[17, 18, 19]])
    print("Base, Other", base_vecs.shape, other_vecs.shape)
    X = base_vecs
    Y = other_vecs
    print("X = \n", X.T.shape)
    print("Y = \n", Y.T.shape)
    R, _, _, _ = np.linalg.lstsq(Y.T, X.T)
    print("LSTSQ X.T x R = Y.T, R = \n", R.shape)

    L = np.dot(Y.T, R).T
    print("L = \n", L)
    
    base_embed.wv.syn0norm = base_embed.wv.syn0 = X
    other_embed.wv.syn0norm = other_embed.wv.syn0 = L   
    return base_embed, other_embed
    '''
    
    #base_vecs = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    #other_vecs = np.array([[11, 12, 13],[14, 15, 16],[17, 18, 19]])
    print("Base, Other", base_vecs.shape, other_vecs.shape)
    X = base_vecs
    Y = other_vecs
    print("X = \n", X.shape)
    print("Y = \n", Y.shape)
    R, _, _, _ = np.linalg.lstsq(Y, X)
    print("LSTSQ X.T x R = Y.T, R = \n", R.shape)

    L = np.dot(Y, R)
    print("L = \n", L.shape)
    
    base_embed.wv.syn0norm = base_embed.wv.syn0 = X
    other_embed.wv.syn0norm = other_embed.wv.syn0 = L   
    return base_embed, other_embed
    
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm
    #base_vecs = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    #other_vecs = np.array([[11, 12, 13],[14, 15, 16],[17, 18, 19]])
    
    m = np.dot(np.transpose(other_vecs), base_vecs)
    print("M = Other^T Dot Base: ", m.shape)
    
    # SVD method from numpy
    u, null, v = np.linalg.svd(m)
    print("SVD of M (U, Null, V): ", u.shape, null.shape, v.shape)
    
    # another matrix operation
    ortho = np.dot(u, v)
    print("U Dot V: ", ortho.shape)
    
    transformed_base_vecs = np.dot(base_vecs, ortho)
    transformed_other_vecs = np.dot(other_vecs, ortho)
    
    print(np.array_equal(base_embed.wv.syn0, transformed_base_vecs))
    print(np.array_equal(other_embed.wv.syn0, transformed_other_vecs))
    
    base_embed.wv.syn0norm = base_embed.wv.syn0 = transformed_base_vecs
    other_embed.wv.syn0norm = other_embed.wv.syn0 = transformed_other_vecs
    
    print(np.array_equal(base_embed.wv.syn0, transformed_base_vecs))
    print(np.array_equal(other_embed.wv.syn0, transformed_other_vecs))
    print("Returning: ", base_embed.wv.syn0.shape, other_embed.wv.syn0.shape)
    print()
    
    print("Base: \n", base_embed.wv.syn0)
    print("Other: \n", other_embed.wv.syn0)
    return base_embed, other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
            -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
            -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    
    if words:
        common_vocab &= set(words)

    print("Align Liberal, Conservative = ", (len(vocab_m1), len(vocab_m2)))

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)

    print("Common Vocabulary = ", len(common_vocab))
    print()
    #print(common_vocab[0:10])
    
    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0#norm
        
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        print("Vector Shape = ", (new_arr.shape, m.wv.syn0.shape))

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
 
        m.index2word = common_vocab       
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, 
                    count=old_vocab_obj.count)
        
        m.wv.vocab = new_vocab
        print("Index2Word, Vocab = ", (len(m.index2word), len(m.wv.vocab)))
        print()

    return (m1, m2)


import sys, gensim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
np.random.seed(7)
#first_word2vec_model = Word2Vec.load('updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
#second_word2vec_model = Word2Vec.load('updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

first_word2vec_model = gensim.models.Word2Vec.load(
        'updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
second_word2vec_model = gensim.models.Word2Vec.load(
        'updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

print(first_word2vec_model.wv.syn0.shape, second_word2vec_model.wv.syn0.shape)
first_aligned_model, second_aligned_model = \
    smart_procrustes_align_gensim(first_word2vec_model, second_word2vec_model)

# Get the vocab for each model
vocab_m1 = set(first_aligned_model.wv.vocab.keys())
vocab_m2 = set(second_aligned_model.wv.vocab.keys())

# Find the common vocabulary
common_vocab = vocab_m1 & vocab_m2
#assert(len(common_vocab) == len(vocab_m1) and len(common_vocab) == len(vocab_m2))
print("Words in Intersection: ", len(common_vocab), len(vocab_m1), len(vocab_m2))
#print(common_vocab)

word = 'lgbt'#'unamerican'#'god'#'freedom'# sys.argv[1]#
topcount = 10#len(common_vocab)# int(sys.argv[2])#100
max_visible = 10
first_word2vec_model = gensim.models.Word2Vec.load(
        'updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
second_word2vec_model = gensim.models.Word2Vec.load(
        'updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

first_neighbors = first_word2vec_model.most_similar(word, topn=topcount)#len(first_word2vec_model.wv.vocab))#topcount)#
first_labels = [entry[0] for entry in first_neighbors if entry[0] in first_aligned_model.wv.vocab]
#first_labels = sorted(first_labels)
first_labels.insert(0, word)
print("liberal model: ", word, len(first_labels), first_labels[0:max_visible])

second_neighbors = second_word2vec_model.most_similar(word, topn=topcount)#len(second_word2vec_model.wv.vocab))#topcount)#
second_labels = [entry[0] for entry in second_neighbors if entry[0] in second_aligned_model.wv.vocab]
#second_labels = sorted(second_labels)
second_labels.insert(0, word)
print("conservative model: ", word, len(second_labels), second_labels[0:max_visible])

#second_labels = first_labels
#first_labels = second_labels
first_vectors = [first_aligned_model[label] for label in first_labels]
second_vectors = [second_aligned_model[label] for label in second_labels]# if label in second_aligned_model.wv.vocab]

#exit()
'''
print("Transformed Space, Common TSNE")
print("------------------------------\n")
labels = first_labels + second_labels
vectors = first_vectors + second_vectors

tsne = TSNE(n_components=2)
data = tsne.fit_transform(vectors)

plt.subplots_adjust(bottom = 0.1)
plt.scatter(data[:, 0]*100, data[:, 1]*100, marker='o')

first_node = True
color = 'cyan'
for label, x, y in zip(first_labels, data[0:len(first_labels), 0], data[0:len(first_labels), 1]):
            
    plt.annotate(
        label,
        xy=(x*100, y*100), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    if first_node:
        color = 'blue'
        first_node = False

first_node = True
color = 'orange'
for label, x, y in zip(second_labels, data[len(first_labels):, 0], data[len(first_labels):, 1]):
    plt.annotate(
        label,
        xy=(x*100, y*100), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    if first_node:
        color = 'red'
        first_node = False

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 9)
plt.show()
'''

print("Transformed Space, Separate PCA")
print("------------------------------\n")
print("Shape of Transformed Space: ", first_word2vec_model.wv.syn0.shape, second_word2vec_model.wv.syn0.shape)
first_vectors = [first_aligned_model[w] for w in common_vocab]
second_vectors = [second_aligned_model[w] for w in common_vocab]

pca = PCA(n_components=2)
first_data = pca.fit_transform(first_vectors)
pca = PCA(n_components=2)
second_data = pca.fit_transform(second_vectors)

first_pca = [first_data[ind, :] for ind,data in enumerate(common_vocab) if data in first_labels]
second_pca = [second_data[ind, :] for ind,data in enumerate(common_vocab) if data in second_labels]

plt.subplots_adjust(bottom = 0.1)
for data,labels,color_node,color_data in \
    [(first_pca, first_labels, 'cyan', 'blue'), (second_pca, second_labels, 'orange', 'red')]:
    data = np.array(data)
    print("shape: ", data.shape)
    print(data)
    #print(labels)
    plt.scatter(data[0:max_visible, 0]*100, data[0:max_visible, 1]*100, marker='o', color=color_data)
    
    first_node = True
    color = color_node
    for label, x, y in zip(labels, data[0:max_visible, 0], data[0:max_visible, 1]):
                
        plt.annotate(
            label,
            xy=(x*100, y*100), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
        if first_node:
            color = color_data
            first_node = False
            #break

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 9)
plt.show()



print("Transformed Space, Common PCA")
print("------------------------------\n")
print("Shape of Transformed Space: ", first_aligned_model.wv.syn0.shape, second_aligned_model.wv.syn0.shape)
first_vectors = [first_aligned_model[w] for w in common_vocab]
second_vectors = [second_aligned_model[w] for w in common_vocab]

total_vectors = first_vectors + second_vectors
#total_vectors = first_aligned_model.wv.syn0 + second_aligned_model.wv.syn0
pca = PCA(n_components=2)
pca_model = pca.fit(total_vectors)
first_data = pca_model.transform(first_vectors)
pca = PCA(n_components=2)
second_data = pca_model.transform(second_vectors)

first_pca = [first_data[ind, :] for ind,data in enumerate(common_vocab) if data in first_labels]
second_pca = [second_data[ind, :] for ind,data in enumerate(common_vocab) if data in second_labels]

plt.subplots_adjust(bottom = 0.1)
for data,labels,color_node,color_data in \
    [(first_pca, first_labels, 'cyan', 'blue'), (second_pca, second_labels, 'orange', 'red')]:
    data = np.array(data)
    print("shape: ", data.shape)
    print(data)
    #print(labels)
    plt.scatter(data[0:max_visible, 0]*100, data[0:max_visible, 1]*100, marker='o', color=color_data)
    
    first_node = True
    color = color_node
    for label, x, y in zip(labels, data[0:max_visible, 0], data[0:max_visible, 1]):
                
        plt.annotate(
            label,
            xy=(x*100, y*100), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
        if first_node:
            color = color_data
            first_node = False
            #break
    
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 9)
plt.show()




print("Transformed Space, Transformed PCA")
print("------------------------------\n")
print("Shape of Transformed Space: ", first_aligned_model.wv.syn0.shape, second_aligned_model.wv.syn0.shape)
first_vectors = [first_aligned_model[w] for w in common_vocab]
second_vectors = [second_aligned_model[w] for w in common_vocab]

total_vectors = first_vectors# + second_vectors
pca = PCA(n_components=2)
pca_model = pca.fit(total_vectors)
first_data = pca_model.transform(first_vectors)
pca = PCA(n_components=2)
second_data = pca_model.transform(second_vectors)

first_pca = [first_data[ind, :] for ind,data in enumerate(common_vocab) if data in first_labels]
second_pca = [second_data[ind, :] for ind,data in enumerate(common_vocab) if data in second_labels]

plt.subplots_adjust(bottom = 0.1)
for data,labels,color_node,color_data in \
    [(first_pca, first_labels, 'cyan', 'blue'), (second_pca, second_labels, 'orange', 'red')]:
    data = np.array(data)
    print("shape: ", data.shape)
    print(data)
    #print(labels)
    plt.scatter(data[0:max_visible, 0]*100, data[0:max_visible, 1]*100, marker='o', color=color_data)
    
    first_node = True
    color = color_node
    for label, x, y in zip(labels, data[0:max_visible, 0], data[0:max_visible, 1]):
                
        plt.annotate(
            label,
            xy=(x*100, y*100), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
        if first_node:
            color = color_data
            first_node = False
            #break
    
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 9)
plt.show()


print("Liberal = ", word, first_aligned_model[word])
print("Conservative = ", word, second_aligned_model[word])
print("Equality = ", np.array_equal(first_aligned_model[word], second_aligned_model[word]))